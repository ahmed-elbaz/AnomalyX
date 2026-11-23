
import json
from typing import TypedDict
from rag_retriever import documents, retrieve_knowledge
import ollama
import pandas as pd
import re
from prompt_interpreter import interpret_prompt
from langgraph.graph import StateGraph, START, END
from evidence_validator import build_evidence_summary
from evidence_validator import build_evidence_summary, describe_evidence
from investigation_tools import (
    get_product_context,
    get_historical_context,
    compare_with_peers,
    get_promotion_context
)
from evidence_validator import (
    build_evidence_summary,
    describe_evidence,
    check_direction_contradictions
)
import time

MAX_TOOL_CALLS = 6
MAX_MISSING_RETRIES = 2

# 1. Shared state
class AgentState(TypedDict):
    product_id: str
    date: str
    messages: list
    executed_tools: list[str]
    tool_call_count: int
    investigation_status: str
    tool_results: dict
    missing_retries: int


# 2. Tool descriptions for Qwen
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_context",
            "description": "Get current product sales, inventory, price and promotion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["product_id", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_context",
            "description": "Get previous product sales, inventory, price and revenue averages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["product_id", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_with_peers",
            "description": "Compare product demand and revenue against same-category products on the same date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["product_id", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_promotion_context",
            "description": "Compare current promotion performance with previous promotion days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["product_id", "date"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "search_investigation_knowledge",
        "description": (
            "Retrieve domain knowledge about possible causes, "
            "investigation steps, and recommended actions. "
            "Use after interpreting factual evidence and forming "
            "an investigation hypothesis, such as stockout or demand drop. "
            "This tool does not provide product-specific facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A specific investigation hypothesis or topic, "
                        "such as 'stockout causes and recommended actions'."
                    )
                }
            },
            "required": ["query"]
        }
    }
}
]


# 3. Python tool registry
TOOL_FUNCTIONS = {
    "get_product_context": get_product_context,
    "get_historical_context": get_historical_context,
    "compare_with_peers": compare_with_peers,
    "get_promotion_context": get_promotion_context
}


# 4. Agent node
def agent_node(state: AgentState):
    start = time.perf_counter()
    response = ollama.chat(
        model="qwen3:8b",
        think=False,
        messages=state["messages"],
        tools=TOOLS,
        options={"num_predict": 350}
    )
    print(f"Agent LLM call: {time.perf_counter() - start:.1f}s"
)
    message = response["message"]

    if message.get("tool_calls"):
        names = [
            call["function"]["name"]
            for call in message["tool_calls"]
        ]
        print("Qwen requested:", names)
    else:
        print("Qwen answer:", message.get("content", ""))

    return {"messages": state["messages"] + [message]}


# 5. Tool node
def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    new_messages = []

    executed_tools = state["executed_tools"].copy()
    tool_call_count = state["tool_call_count"]

    # Store previously successful results for reuse.
    cached_results = state["tool_results"].copy()

    for call in last_message.get("tool_calls", []):
        name = call["function"]["name"]
        arguments = call["function"].get("arguments", {})

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        if not isinstance(arguments, dict):
            arguments = {}

        # Analytical tools have fixed inputs from trusted state.
        if name == "search_investigation_knowledge":
            query = arguments.get("query", "")
            if not isinstance(query, str):
                query = ""
            cache_key = f"{name}:{query.strip().lower()}"
        else:
            cache_key = f"{name}:{state['product_id']}:{state['date']}"

        if cache_key in cached_results:
            result = cached_results[cache_key]
            print(f"Cached result reused: {name}")

        elif tool_call_count >= MAX_TOOL_CALLS:
            result = {"error": "Tool-call budget exhausted."}
            print(f"Budget exceeded: {name}")

        else:
            try:
                if name == "search_investigation_knowledge":
                    if not query.strip():
                        result = {"error": "A knowledge search query is required."}
                    else:
                        matches = retrieve_knowledge(
                            query=query,
                            documents=documents,
                            top_k=1
                        )
                        result = [
                            {"source": item["source"], "text": item["text"]}
                            for item in matches
                        ]

                elif name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[name](
                        state["product_id"],
                        state["date"]
                    )

                else:
                    result = {"error": f"Unknown tool: {name}"}

            except Exception as exc:
                result = {"error": str(exc)}

            # Count actual attempted executions, not cache hits.
            if name in TOOL_FUNCTIONS or name == "search_investigation_knowledge":
                tool_call_count += 1

            # Only successful results become evidence.
            if not isinstance(result, dict) or "error" not in result:
                cached_results[cache_key] = result
                if name not in executed_tools:
                    executed_tools.append(name)

            print(f"Tool executed: {name}")
            print("Result:", result)

        new_messages.append({
            "role": "tool",
            "tool_name": name,
            "content": json.dumps(result, default=str)
        })

    return {
        "messages": state["messages"] + new_messages,
        "executed_tools": executed_tools,
        "tool_call_count": tool_call_count,
        "tool_results": cached_results
    }


def validate_investigation(state):
    required = {
        "get_product_context",
        "get_historical_context",
        "search_investigation_knowledge"
    }

    # Retrieve product evidence
    product_key = (
        f"get_product_context:{state['product_id']}:{state['date']}"
    )

    product = state["tool_results"].get(product_key)

    # Require promotion investigation when relevant
    if isinstance(product, dict):
        if product.get("promotion") == 1 and product.get("demand_ratio", 1) < 1:
            required.add("get_promotion_context")

    # Identify missing required tools
    missing = required - set(state["executed_tools"])

    return missing


def request_missing_node(state: AgentState):
    missing = validate_investigation(state)

    feedback = {
        "role": "user",
        "content": (
            "The investigation is incomplete. "
            f"You MUST call these missing tools now: {sorted(missing)}. "
            "Do not produce a final answer yet."
        )
    }

    return {
        "messages": state["messages"] + [feedback],
        "missing_retries": state["missing_retries"] + 1
    }


def collect_missing_promotion_node(state: AgentState):
    product_id = state["product_id"]
    date = state["date"]

    tool_call_count = state["tool_call_count"] + 1

    try:
        if state["tool_call_count"] >= MAX_TOOL_CALLS:
            raise RuntimeError("Tool-call budget exhausted.")

        result = get_promotion_context(product_id, date)

        if not isinstance(result, dict) or not result or "error" in result:
            raise ValueError(f"Invalid promotion evidence: {result}")

    except Exception as exc:
        print(f"Promotion collection failed: {exc}")
        return {
            "tool_call_count": tool_call_count,
            "investigation_status": "incomplete",
            "messages": state["messages"] + [{
                "role": "user",
                "content": f"Promotion evidence collection failed: {exc}"
            }]
        }

    cache_key = f"get_promotion_context:{product_id}:{date}"

    executed_tools = state["executed_tools"].copy()
    executed_tools.append("get_promotion_context")

    tool_results = state["tool_results"].copy()
    tool_results[cache_key] = result

    print("Automatically collected required promotion context:", result)

    return {
        "executed_tools": executed_tools,
        "tool_call_count": tool_call_count,
        "tool_results": tool_results,
        "messages": state["messages"] + [{
            "role": "user",
            "content": f"Required promotion evidence: {json.dumps(result, default=str)}"
        }]
    }


def incomplete_node(state):
    print("Investigation stopped: tool budget exhausted.")
    return {"investigation_status": "incomplete"}


def final_answer_node(state: AgentState):


    product_key = (
        f"get_product_context:{state['product_id']}:{state['date']}"
    )

    history_key = (
        f"get_historical_context:{state['product_id']}:{state['date']}"
    )

    product = state["tool_results"].get(product_key)
    history = state["tool_results"].get(history_key)


    peer_key = f"compare_with_peers:{state['product_id']}:{state['date']}"
    promo_key = f"get_promotion_context:{state['product_id']}:{state['date']}"

    peer = state["tool_results"].get(peer_key)
    promo = state["tool_results"].get(promo_key)

    additional_evidence = {}

    if isinstance(peer, dict) and "error" not in peer:
        additional_evidence["peer_comparison"] = {
            "product_demand_ratio": peer.get("product_demand_ratio"),
            "peer_avg_demand_ratio": peer.get("peer_avg_demand_ratio"),
            "product_revenue_ratio": peer.get("product_revenue_ratio"),
            "peer_avg_revenue_ratio": peer.get("peer_avg_revenue_ratio")
        }

    if isinstance(promo, dict) and "error" not in promo:
        additional_evidence["promotion_comparison"] = {
            "current_units_sold": promo.get("current_units_sold"),
            "previous_promo_avg_units_sold": promo.get("previous_promo_avg_units_sold"),
            "previous_promo_days": promo.get("previous_promo_days"),
            "current_demand_ratio": promo.get("current_demand_ratio"),
            "previous_promo_avg_demand_ratio": promo.get("previous_promo_avg_demand_ratio")
        }



    if not isinstance(product, dict) or not isinstance(history, dict):
        raise ValueError("Required evidence is missing or invalid.")

    evidence_summary = build_evidence_summary(product, history)
    evidence_descriptions = describe_evidence(evidence_summary)

    print("VERIFIED EVIDENCE:", evidence_summary)

    final_instruction = {
        "role": "user",
        "content": (
            f"Verified numerical evidence: {evidence_summary}. "
            f"Verified peer and promotion evidence: {additional_evidence}. "
            f"Verified evidence statements: {' '.join(evidence_descriptions)} "
            "Never reverse the direction of these changes. "
            "These percentage changes were calculated by Python. "
            "Use them accurately in your report. "
            "Use peer comparison evidence when available to determine "
            "whether the anomaly is product-specific or category-wide. "
            "Use promotion context when available to evaluate whether "
            "the active promotion performed better or worse than previous promotions. "
            "Explicitly mention the number of previous promotion days "
            "when interpreting promotion performance. "
            "Do not claim that a price decrease caused a demand decline. "
            "Do not invent causes or treat hypotheses as verified facts. "
            "The evidence collection phase is finished. "
            "Using only the evidence already collected, produce the final "
            "investigation report with these four sections: "
            "1. Observed anomaly, "
            "2. Most likely explanation supported by evidence, "
            "3. What remains unknown, "
            "4. Recommended next action. "
            "Distinguish verified facts from possible causes. "
            "Do not invent evidence. Keep the report concise. "
            "When peer or promotion evidence is available, include the actual "
            "comparison numbers in the report, not just qualitative descriptions. "
            "A lower price occurring alongside lower demand does not establish "
            "that the price reduction caused the demand decline. "
            "Do not suggest that the price reduction caused or contributed to "
            "the decline without independent supporting evidence. "

            "Evidence rules: "
            "1. Prioritize numerical evidence from executed tools. "
            "2. Separate observed facts from possible explanations. "
            "3. RAG guidance is general knowledge, not proof of a specific cause. "
            "4. Do not present competitor activity or customer behavior "
            "as likely causes without supporting evidence. "
            "5. Explicitly discuss significant price differences "
            "compared with historical averages. "
            "6. Never claim causality without supporting evidence. "
            "7. Clearly state what remains unknown."
        )
    }

    final_messages = state["messages"].copy()

    last_message = final_messages[-1]

    if last_message.get("tool_calls"):
        final_messages.append({
            "role": "tool",
            "content": (
                "Tool execution was skipped because the investigation "
                "budget is exhausted. Use the evidence already collected."
            )
        })

    start = time.perf_counter()

    # Use existing history plus final reporting instruction.
    # The instruction is not added permanently to history.
    response = ollama.chat(
        model="qwen3:8b",
        think=False,
        messages=final_messages + [final_instruction],
        options={"num_predict": 450}
    )
    print(f"Final report LLM call: {time.perf_counter() - start:.1f}s")
    message = response["message"]
    content = message.get("content", "").strip()
    done_reason = response.get("done_reason")



    # Check final report against verified numerical evidence
    contradictions = check_direction_contradictions(
        content, evidence_summary
    )

    if contradictions:
        print("EVIDENCE CONTRADICTIONS:", contradictions)
        print("Retrying final report once...")

        retry_instruction = {
            "role": "user",
            "content": (
                "Your previous report contradicted verified numerical evidence. "
                f"Contradictions: {contradictions}. "
                f"Correct evidence: {' '.join(evidence_descriptions)} "
                "Rewrite the full four-section report. "
                "Preserve the correct facts and do not invent causes."
            )
        }

        start = time.perf_counter()

        response = ollama.chat(
            model="qwen3:8b",
            think=False,
            messages=final_messages + [
                final_instruction,
                message,
                retry_instruction
            ],
            options={"num_predict": 450}
        )

        print(f"Final report RETRY: {time.perf_counter() - start:.1f}s")

        message = response["message"]
        content = message.get("content", "").strip()
        done_reason = response.get("done_reason")

        # Validate the revised report
        contradictions = check_direction_contradictions(
            content, evidence_summary
        )


    print("\nFINAL REPORT GENERATED")
    print("Stop reason:", done_reason)

    # Accept only a nonempty response that stopped normally.
    valid = (
    bool(content)
    and done_reason == "stop"
    and not contradictions
    )

    return {
        "messages": final_messages + [final_instruction, message],
        "investigation_status": "completed" if valid else "incomplete"
    }




# 6. Conditional routing


def route_next_step(state):
    last_message = state["messages"][-1]

    missing = validate_investigation(state)
    budget_exhausted = state["tool_call_count"] >= MAX_TOOL_CALLS

    print("\nROUTING CHECK:")
    print("Missing tools:", missing)
    print("Tool calls used:", state["tool_call_count"])

    # Required evidence is collected.
    if not missing:
        return "final_answer"

    # Cannot continue without exceeding budget.
    if budget_exhausted:
        return "incomplete"

    # Qwen requested tools.
    if last_message.get("tool_calls"):
        return "tools"

    # Qwen did not request tools; ask again, within retry limit.
    if state["missing_retries"] >= MAX_MISSING_RETRIES:
        return "incomplete"

    return "request_missing"




# 7. Build graph
builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("request_missing", request_missing_node)
builder.add_node("incomplete", incomplete_node)
builder.add_node("final_answer", final_answer_node)
builder.add_node(
    "collect_missing_promotion",
    collect_missing_promotion_node
)

builder.add_conditional_edges(
    "agent",
    route_next_step,
    {
        "tools": "tools",
        "request_missing": "request_missing",
        "final_answer": "final_answer",
        "incomplete": "incomplete"
    }
)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "tools",
    lambda state: (
        "collect_missing_promotion"
        if validate_investigation(state) == {"get_promotion_context"}
        else (
            "final_answer"
            if not validate_investigation(state)
            else "agent"
        )
    ),
    {
        "collect_missing_promotion": "collect_missing_promotion",
        "final_answer": "final_answer",
        "agent": "agent"
    }
)
builder.add_conditional_edges(
    "collect_missing_promotion",
    lambda state: (
        "incomplete"
        if state["investigation_status"] == "incomplete"
        else "final_answer"
    ),
    {
        "incomplete": "incomplete",
        "final_answer": "final_answer"
    }
)
builder.add_edge("request_missing", "agent")
builder.add_edge("incomplete", END)
builder.add_edge("final_answer", END)

graph = builder.compile()



def run_investigation(product_id, date):
    product_id = str(product_id).upper()
    date = str(date)

    result = graph.invoke(
        {
            "product_id": product_id,
            "date": date,
            "executed_tools": [],
            "tool_call_count": 0,
            "missing_retries": 0,
            "investigation_status": "investigating",
            "tool_results": {},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are AnomalyX, an e-commerce anomaly investigation agent. "
                        "Start with get_product_context. "
                        "Investigate relevant historical and peer behavior. "
                        "Use promotion context only when relevant. "
                        "Interpret factual evidence and form a supported hypothesis. "
                        "Then call search_investigation_knowledge with a specific "
                        "hypothesis-based query to retrieve investigation guidance. "
                        "Do not use RAG to classify raw numerical evidence. "
                        "Do not repeat unnecessary tool calls. "
                        "Your final answer must include: "
                        "1. Observed anomaly, "
                        "2. Most likely explanation supported by evidence, "
                        "3. What remains unknown, "
                        "4. Recommended next action. "
                        "Never present general knowledge as verified product-specific facts. "
                        "Keep the final answer concise."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Investigate product {product_id} on {date}."
                    )
                }
            ]
        },
        config={"recursion_limit": 20}
    )

    return result


if __name__ == "__main__":

    mode = input(
        "Choose investigation mode (1=LOF, 2=User): "
    ).strip()

    if mode == "1":
        anomalies = pd.read_csv("lof_detected_anomalies.csv")
        selected = anomalies.iloc[0]

        detected_product_id = str(selected["product_id"])
        detected_date = str(selected["date"])

    elif mode == "2":
        user_prompt = input("Enter investigation request: ").strip()
        extracted = interpret_prompt(user_prompt)

        detected_product_id = extracted.get("product_id")
        detected_date = extracted.get("date")

        if not detected_product_id:
            raise ValueError("Please specify which product to investigate.")

        if not detected_date:
            detected_date = input(
                f"Please provide the investigation date for "
                f"{detected_product_id} (YYYY-MM-DD): "
            ).strip()

        detected_product_id = str(detected_product_id).upper()
        detected_date = str(detected_date)

        features = pd.read_csv("ecommerce_features.csv")

        exists = (
            (features["product_id"] == detected_product_id)
            & (features["date"] == detected_date)
        ).any()

        if not exists:
            raise ValueError("Product/date not found in the dataset.")

    else:
        raise ValueError("Choose mode 1 or 2.")

    print("Selected investigation:", detected_product_id, detected_date)

    result = run_investigation(
        detected_product_id,
        detected_date
    )

    print("\nINVESTIGATION TRACKING:")
    print("Executed tools:", result["executed_tools"])
    print("Tool call count:", result["tool_call_count"])
    print("Investigation status:", result["investigation_status"])
    print("\nFINAL ANSWER:")

    if result["investigation_status"] == "completed":
        print(result["messages"][-1].get("content", ""))
    else:
        print("No valid final report generated. Investigation incomplete.")
