"""
L05 — 记忆：让 Agent 记住上下文
================================
实现一个带记忆的多轮对话 Agent，演示三种窗口管理策略：
    ① 多轮对话（Agent 记住前面说过的话）
    ② 三种窗口管理对比：全保留 / 截断 / 摘要压缩
    ③ 演示"截断导致失忆"

运行：python agent-lessons/05_memory/code.py
"""
from __future__ import annotations

import json
import chromadb
import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

CHAT_MODEL = "glm-4-flash"  # 想免费可换 "glm-4-flash"


def create_client() -> ZhipuAI:
    load_dotenv()
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("xxxx"):
        raise RuntimeError("请先在 .env 里配置 ZHIPUAI_API_KEY")
    return ZhipuAI(api_key=api_key)


def chat(client: ZhipuAI, messages: list) -> str:
    """基础对话：传入完整 messages（含历史），返回回答。"""
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return resp.choices[0].message.content


# ════════════════════════════════════════════════════════════
# 工具函数：估算 token（教学用粗略估算）
# ════════════════════════════════════════════════════════════
def estimate_tokens(messages: list) -> int:
    """粗略估算 messages 的 token 数。

    中文约 1 字 ≈ 1.5 token，英文约 4 字符 ≈ 1 token。
    这里用最简单的估算：总字符数 / 2（教学够用，真实用 tokenizer）。
    """
    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    return total_chars // 2


# ════════════════════════════════════════════════════════════
# 策略 1：全保留（不做任何处理）
# ════════════════════════════════════════════════════════════
def keep_all(messages: list) -> list:
    """全保留：原样返回所有消息。"""
    return messages.copy()


# ════════════════════════════════════════════════════════════
# 策略 2：截断（滑动窗口，只保留最近 N 轮）
# ════════════════════════════════════════════════════════════
def truncate_messages(messages: list, keep_last: int = 6) -> list:
    """截断：保留 system 消息 + 最近 keep_last 条非 system 消息。

    keep_last 是"消息条数"（一轮对话 = 用户+助手 = 2条）。
    所以 keep_last=6 约等于保留最近 3 轮对话。
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]
    kept = other_msgs[-keep_last:] if keep_last > 0 else []
    return system_msgs + kept


# ════════════════════════════════════════════════════════════
# 策略 3：摘要压缩（用模型把旧历史压成摘要）
# ════════════════════════════════════════════════════════════
def summarize_history(client: ZhipuAI, old_messages: list) -> str:
    """用模型把一段旧的对话历史压缩成摘要。"""
    # 把旧历史拼成纯文本
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in old_messages if m.get("content")
    )
    prompt = (
        "请把下面这段对话历史压缩成一段简洁的摘要，保留关键信息"
        "（用户的名字、偏好、重要结论、未解决的问题等），不超过 100 字：\n\n"
        f"{history_text}"
    )
    return "[之前的对话摘要] " + chat(client, [{"role": "user", "content": prompt}])

# 上面是每轮都压缩，而是当消息超过阈值时才压缩一次
class MemoryManager:
    def __init__(self, threshold=10, keep_recent=4):
        self.threshold = threshold  # 超过这个数才压缩
        self.keep_recent = keep_recent

    def manage(self, client, messages):
        other = [m for m in messages if m["role"] != "system"]
        if len(other) > self.threshold:
            print(f"  [MemoryManager] 消息数 {len(other)} > {self.threshold}，触发压缩")
            return compress_with_summary(client, messages, self.keep_recent)
        return messages.copy()  # 没超阈值，不压缩


def compress_with_summary(client: ZhipuAI, messages: list, keep_recent: int = 4) -> list:
    """摘要压缩：把旧历史压成一条摘要 + 保留最近几轮原样。

    keep_recent：最近多少条消息原样保留（不压缩）。
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    # 如果消息不多，不用压缩
    if len(other_msgs) <= keep_recent:
        return messages.copy()

    # 分成"要压缩的旧部分" + "保留的近期部分"
    to_compress = other_msgs[:-keep_recent]
    recent = other_msgs[-keep_recent:]

    # 把旧部分压成摘要
    summary = summarize_history(client, to_compress)
    summary_msg = {"role": "user", "content": summary}

    return system_msgs + [summary_msg] + recent


# ════════════════════════════════════════════════════════════
# 实验 1：多轮对话（证明 Agent 有记忆）
# ════════════════════════════════════════════════════════════
def demo_multi_turn(client: ZhipuAI):
    """演示多轮对话——Agent 能记住前面说过的信息。"""
    print("\n" + "═" * 60)
    print("实验 1：多轮对话（Agent 记住上下文）")
    print("═" * 60)

    # messages 就是 Agent 的"记忆"，每轮都带着完整历史
    messages = [{"role": "system", "content": "你是一个友好的助手。"}]

    # 模拟 3 轮对话
    conversations = [
        "我叫张三，我喜欢蓝色。",
        "我今天心情不太好。",
        "我还记得我跟你说过我的名字和喜欢的颜色吗？",  # 测试记忆
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n第 {i} 轮：")
        print(f"  🙋 用户：{user_input}")
        messages.append({"role": "user", "content": user_input})

        reply = chat(client, messages)
        print(f"  🤖 助手：{reply}")
        messages.append({"role": "assistant", "content": reply})

    print("\n👉 观察：第 3 轮 Agent 应该能答出'张三'和'蓝色'，因为历史在 messages 里。")
    print("   💡 模型自己没记忆，全靠我们把历史每次都传给它。")


# ════════════════════════════════════════════════════════════
# 实验 2：三种窗口管理策略对比
# ════════════════════════════════════════════════════════════
def demo_strategies(client: ZhipuAI):
    """模拟一个长对话，对比三种策略的 token 数和信息保留。"""
    print("\n\n" + "═" * 60)
    print("实验 2：三种窗口管理策略对比")
    print("═" * 60)

    # 模拟一个 10 轮的长对话历史
    long_messages = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "我叫张三。"},
        {"role": "assistant", "content": "你好，张三！"},
        {"role": "user", "content": "我喜欢蓝色。"},
        {"role": "assistant", "content": "蓝色是很棒的颜色。"},
        {"role": "user", "content": "北京今天天气怎么样？"},
        {"role": "assistant", "content": "我不知道实时天气，但可以帮你查。"},
        {"role": "user", "content": "算了，帮我算 12*34。"},
        {"role": "assistant", "content": "12*34=408。"},
        {"role": "user", "content": "谢谢，再见。"},
        {"role": "assistant", "content": "再见张三！"},
        {"role": "user", "content": "对了，我还想问..."},
        {"role": "assistant", "content": "请问吧。"},
    ]

    print(f"\n原始历史：{len(long_messages)} 条消息")

    # 策略 1：全保留
    m1 = keep_all(long_messages)
    print(f"\n【全保留】{len(m1)} 条消息，约 {estimate_tokens(m1)} tokens")
    print(f"  → 信息完整，但对话越长 token 越多（贵且慢）")

    # 策略 2：截断（保留最近 6 条 = 3 轮）
    m2 = truncate_messages(long_messages, keep_last=6)
    print(f"\n【截断 keep_last=6】{len(m2)} 条消息，约 {estimate_tokens(m2)} tokens")
    print(f"  保留的内容：")
    for m in m2:
        if m["role"] != "system":
            print(f"    {m['role']}: {m['content'][:30]}")
    print(f"  → 省了 token，但'我叫张三''我喜欢蓝色'被丢了（失忆）")

    # 策略 3：摘要压缩
    print(f"\n【摘要压缩】正在用模型压缩旧历史...")
    # m3 = compress_with_summary(client, long_messages, keep_recent=4)
    m3 = MemoryManager(threshold=10, keep_recent=4).manage(client, long_messages)
    print(f"{len(m3)} 条消息，约 {estimate_tokens(m3)} tokens")
    print(f"  保留的内容：")
    for m in m3:
        content = (m.get("content") or "")[:60]
        print(f"    {m['role']}: {content}")
    print(f"  → 旧历史被压成摘要（保留了'张三''蓝色'等关键信息），最近几轮原样保留")


# ════════════════════════════════════════════════════════════
# 实验 3：截断导致"失忆"
# ════════════════════════════════════════════════════════════
def demo_forgetting(client: ZhipuAI):
    """演示截断策略下，Agent 会忘记早期信息。"""
    print("\n\n" + "═" * 60)
    print("实验 3：截断导致'失忆'")
    print("═" * 60)

    # 一个对话：第1轮说了名字，中间聊了很多别的，最后问名字
    messages = [{"role": "system", "content": "你是助手。请简短回答。"}]

    rounds = [
        "我叫李四，请记住我的名字。",
        "帮我算 1+1。",
        "帮我算 2+2。",
        "帮我算 3+3。",
    ]
    for r in rounds:
        messages.append({"role": "user", "content": r})
        messages.append({"role": "assistant", "content": chat(client, messages)})

    # 现在问名字
    question = "我叫什么名字？"
    print(f"\n用户在第 5 轮问：{question}")

    # 用全保留策略
    full_msgs = messages + [{"role": "user", "content": question}]
    print(f"\n【全保留】回答：{chat(client, full_msgs)}")

    # 用截断策略（只留最近 4 条 = 2 轮，把"我叫李四"截掉了）
    trunc_msgs = truncate_messages(messages, keep_last=4) + [{"role": "user", "content": question}]
    # trunc_msgs = truncate_messages(messages, keep_last=8) + [{"role": "user", "content": question}]

    print(f"【截断 keep_last=4】回答：{chat(client, trunc_msgs)}")

    print("\n👉 观察：全保留能答出'李四'；截断可能答不出（因为'我叫李四'被截掉了）。")
    print("   这就是截断的代价——省了 token，但丢了早期信息。摘要策略能缓解。")

# ════════════════════════════════════════════════════════════
# 实验 4：交互式
# ════════════════════════════════════════════════════════════
def interactive_mode(client):
    messages = [{"role": "system", "content": "你是友好的助手。"}]
    print("输入问题聊天，输 exit 退出。")
    while True:
        user = input("🙋 ").strip()
        if user.lower() in ("exit", "quit"): break
        messages.append({"role": "user", "content": user})
        reply = chat(client, messages)
        print(f"🤖 {reply}")
        messages.append({"role": "assistant", "content": reply})


# ════════════════════════════════════════════════════════════
# 实验 5：长期记忆（对历史对话做 RAG）
# ════════════════════════════════════════════════════════════
def build_long_memory():
    """初始化持久化向量库（存到 ./memory_db 目录，重启不丢）"""
    chroma_client = chromadb.PersistentClient(path="./memory_db")
    return chroma_client.get_or_create_collection(name="chat_history")

def save_turn(collection, user_msg: str, assistant_msg: str,
              turn_id: int, session_id: str = "default"):
    """把一轮对话存进向量库"""
    doc = f"用户：{user_msg}\n助手：{assistant_msg}"
    collection.add(
        documents=[doc],
        metadatas=[{"session_id": session_id, "turn": turn_id}],
        ids=[f"{session_id}-{turn_id}"],   # id 必须唯一
    )

def retrieve_memory(collection, query: str, n_results: int = 3) -> str:
    """用当前问题检索相关历史"""
    count = collection.count()
    if count == 0:
        return ""
    n = min(n_results, count)              # 防止 n_results > 文档数报错
    results = collection.query(query_texts=[query], n_results=n)
    docs = results.get("documents", [[]])[0]
    return "\n---\n".join(docs) if docs else ""

def interactive_with_long_memory(client):
    """带长期记忆的交互式对话"""
    collection = build_long_memory()
    messages = [{"role": "system", "content": "你是友好的助手。"}]
    turn_id = 0
    print("输入问题聊天，输 exit 退出。（长期记忆已开启）")
    while True:
        user = input("🙋 ").strip()
        if user.lower() in ("exit", "quit"):
            break

        # 1) 用当前问题检索相关历史
        memory = retrieve_memory(collection, user, n_results=3)

        # 2) 把检索到的历史注入 system prompt
        system_content = "你是友好的助手。"
        if memory:
            system_content += (
                "\n\n以下是用户过去的相关对话，请参考这些信息回答：\n" + memory
            )

        # 3) 本次请求 = 新 system + 短期历史 + 当前问题
        current_messages = [{"role": "system", "content": system_content}]
        current_messages += messages[1:]          # 跳过旧 system
        current_messages.append({"role": "user", "content": user})

        reply = chat(client, current_messages)
        print(f"🤖 {reply}")

        # 4) 更新短期记忆
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": reply})

        # 5) 写入长期记忆
        save_turn(collection, user, reply, turn_id)
        turn_id += 1


def main():
    print("=" * 60)
    print("L05 — 记忆：让 Agent 记住上下文")
    print("=" * 60)

    client = create_client()

    # demo_multi_turn(client)
    # demo_strategies(client)
    # demo_forgetting(client)
    # interactive_mode(client)
    interactive_with_long_memory(client)
    print("\n" + "=" * 60)
    print("完成！记忆 = messages 列表管理。三种策略各有取舍。")
    print("💡 长期记忆 = 对对话历史做 RAG（复用你 RAG 课的知识）。")
    print("=" * 60)


if __name__ == "__main__":
    main()
