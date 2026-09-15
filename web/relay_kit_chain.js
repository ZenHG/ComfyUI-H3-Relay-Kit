// H3 Relay Kit · Chain 前端
// 在 H3RelayChain 节点上提供按钮：Run / Approve / 连跑 / Stop / Reset。
// 作用：自动推进同一张图里「latent 桥 + 落盘」两个节点的 stage_index，
// 免去每段手动改两个数字。
//
// 找节点规则：优先取与本 Chain 节点**同一个分组框**里的桥+落盘；
// 没有分组就全图找唯一的一对；找到多对则提示先分组。
// 本文件行为对标 H3-Motion-Context 的 Chain 思路，代码为独立实现。

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function nodeCenter(n) {
    return [n.pos[0] + (n.size?.[0] ?? 0) / 2, n.pos[1] + (n.size?.[1] ?? 0) / 2];
}

function inGroup(n, g) {
    const [cx, cy] = nodeCenter(n);
    const b = g.bounding;
    return cx >= b[0] && cx <= b[0] + b[2] && cy >= b[1] && cy <= b[1] + b[3];
}

function findMemberNodes(chainNode, typeName) {
    const all = app.graph._nodes.filter((n) => n.type === typeName && n.mode === 0);
    if (!all.length) return [];
    // 与 Chain 同分组的优先
    for (const g of app.graph.groups ?? []) {
        const members = all.filter((n) => inGroup(n, g));
        const inIt = inGroup(chainNode, g);
        if (inIt && members.length) return members;
    }
    return all; // 没有分组兜底：全图
}

function setStatus(chainNode, text) {
    const w = (chainNode.widgets ?? []).find((x) => x.name === "status");
    if (w) {
        w.value = text;
        chainNode.setDirtyCanvas?.(true, true);
    } else {
        // 后端没声明 status widget 时，提示就只进控制台、界面上看不到。
        // 这里显式 warn 一次，免得"按钮点了没反应"变成无声故障。
        console.warn(
            "[H3 Relay Chain] 节点上没有 status widget（后端 nodes.py 的 H3RelayChain.INPUT_TYPES " +
                "应声明一个可选的 status: STRING）——状态只写进控制台：" + text
        );
    }
    console.info("[H3 Relay Chain]", text);
}

function getStage(n) {
    const w = (n.widgets ?? []).find((x) => x.name === "stage_index");
    return w ? Math.max(0, Math.round(w.value ?? 0)) : 0;
}

function setStage(n, v) {
    const w = (n.widgets ?? []).find((x) => x.name === "stage_index");
    if (w) w.value = Math.max(0, Math.round(v));
}

function findPair(chainNode) {
    const bridges = findMemberNodes(chainNode, "H3RelayMotionContext");
    const saves = findMemberNodes(chainNode, "H3RelayLatentSave");
    if (bridges.length !== 1 || saves.length !== 1) {
        setStatus(
            chainNode,
            `⚠ 找到 桥×${bridges.length} / 落盘×${saves.length}，需要恰好各 1 个。` +
                `请把 Chain、桥、落盘放进同一个分组框（右键 → 添加分组）。`
        );
        return null;
    }
    return { bridge: bridges[0], save: saves[0] };
}

async function queuePrompt(chainNode) {
    try {
        await app.queuePrompt(0, 1);
        setStatus(chainNode, "已排队，采样中…");
    } catch (err) {
        setStatus(chainNode, "⚠ 排队失败：" + err.message);
        throw err;
    }
}

app.registerExtension({
    name: "H3RelayKit.Chain",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3RelayChain") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);
            const node = this;
            // sawMine：本轮执行里是否真的跑过本组桥/落盘。executing(null) 是全局事件，
            // 多个 Chain 组共存时，别的组跑完一轮不能推进本组的段号。
            const state = { mode: "idle", remaining: 0, sawMine: false };

            const stepDone = () => {
                if (state.mode !== "chain") return;
                if (!state.sawMine) return; // 本轮没执行过本组的桥/落盘 → 别的组的收尾，忽略
                state.sawMine = false;
                const pair = findPair(node);
                if (!pair) {
                    state.mode = "idle";
                    return;
                }
                const next = getStage(pair.bridge) + 1;
                setStage(pair.bridge, next);
                setStage(pair.save, next);
                const segW = (node.widgets ?? []).find((x) => x.name === "segments");
                const seg = Math.round(segW?.value ?? 0);
                const infinite = seg <= 0;
                const more = infinite || state.remaining > 1;
                if (state.remaining > 0) state.remaining -= 1;
                if (!more) {
                    state.mode = "idle";
                    setStatus(node, `✅ 连跑结束，当前段号 ${next}（已落盘，可直接继续 Approve）。`);
                    return;
                }
                setStatus(node, `连跑中：第 ${next + 1} 段排队…（剩余 ${infinite ? "∞" : state.remaining}）`);
                queuePrompt(node).catch(() => (state.mode = "idle"));
            };

            api.addEventListener("executing", ({ detail }) => {
                if (detail === null) { stepDone(); return; }
                // 记录「本组节点确实在这轮执行」：只有跑过本组桥/落盘，收尾才归本组
                if (detail.node != null && state.mode === "chain") {
                    const id = Number(detail.node);
                    const pair = findPair(node);
                    if (pair && (id === pair.bridge.id || id === pair.save.id)) {
                        state.sawMine = true;
                    }
                }
            });
            api.addEventListener("execution_error", () => {
                if (state.mode === "chain") {
                    state.mode = "idle";
                    setStatus(node, "⚠ 执行出错，连跑已停（stage_index 保持当前值，可直接重跑）。");
                }
            });
            api.addEventListener("execution_success", () => {
                // 成功事件先于 executing(null) 到达时无需处理；此处仅兜底日志
            });

            node.addWidget("button", "▶ Run（按当前段号跑一次）", null, async () => {
                const pair = findPair(node);
                if (!pair) return;
                setStatus(node, `Run：stage=${getStage(pair.bridge)} 排队中…`);
                state.mode = "idle";
                await queuePrompt(node);
            });

            node.addWidget("button", "✔ Approve（段号+1 并跑下一段）", null, async () => {
                const pair = findPair(node);
                if (!pair) return;
                const next = getStage(pair.bridge) + 1;
                setStage(pair.bridge, next);
                setStage(pair.save, next);
                setStatus(node, `Approve：段号推进到 ${next}，排队中…`);
                state.mode = "idle";
                await queuePrompt(node);
            });

            node.addWidget("button", "⏩ 连跑（按 segments 自动循环）", null, async () => {
                const pair = findPair(node);
                if (!pair) return;
                const segW = (node.widgets ?? []).find((x) => x.name === "segments");
                const seg = Math.round(segW?.value ?? 0);
                state.mode = "chain";
                state.remaining = seg;
                setStatus(node, `连跑开始：segments=${seg <= 0 ? "∞" : seg}，首段排队中…`);
                await queuePrompt(node);
            });

            node.addWidget("button", "⏹ Stop（本轮跑完即停）", null, () => {
                if (state.mode === "chain") {
                    state.mode = "idle";
                    setStatus(node, "已请求停止：当前采样跑完后不再推进。");
                } else {
                    setStatus(node, "当前没有在连跑。");
                }
            });

            node.addWidget("button", "↺ Reset（段号归 0，从第 1 段重来）", null, () => {
                const pair = findPair(node);
                if (!pair) return;
                setStage(pair.bridge, 0);
                setStage(pair.save, 0);
                setStatus(node, "已归 0：下一段将作为第 1 段（不续接，只落盘）。");
            });

            return r;
        };
    },
});
