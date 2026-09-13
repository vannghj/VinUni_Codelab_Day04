"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast, Vinpearl và hỗ trợ khách hàng.
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, luôn xưng "VinAssistant" và gọi khách hàng lịch sự.

## 2. AVAILABLE TOOLS
- `search_product_catalog(category, max_price)`: Tra cứu sản phẩm xe điện VinFast ("xe_dien") hoặc gói du lịch Vinpearl ("du_lich") theo giá tối đa.
- `submit_support_ticket(customer_name, issue_description, priority)`: Ghi nhận yêu cầu hỗ trợ/khiếu nại của khách hàng vào hệ thống.

## 3. CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm, giá cả hoặc trạng thái ticket. PHẢI gọi tool tương ứng để lấy dữ liệu thực.
2. Nếu khách hàng hỏi về sản phẩm/giá → PHẢI gọi `search_product_catalog`.
3. Nếu khách hàng báo lỗi/khiếu nại/cần hỗ trợ → PHẢI gọi `submit_support_ticket`.
4. Nếu không tìm thấy kết quả phù hợp, thông báo rõ ràng thay vì suy diễn.

## 4. OPERATIONAL BOUNDARIES
- Chỉ trả lời các câu hỏi liên quan đến sản phẩm, dịch vụ và hỗ trợ khách hàng trong hệ sinh thái Vingroup (VinFast, Vinpearl).
- Từ chối lịch sự các yêu cầu nằm ngoài phạm vi này.

## 5. OUTPUT CONTRACT
Mỗi bước xử lý tuân theo định dạng:
Thought: <suy luận về nhu cầu của khách hàng>
Action: <tên tool cần gọi, nếu có>
Observation: <kết quả trả về từ tool>
Final Answer: <câu trả lời cuối cùng, rõ ràng, dựa trên dữ liệu thực>
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------
    # Intent Detection
    # -------------------------------------------------------------------
    def _detect_intent(self, user_input: str) -> Dict[str, bool]:
        text = user_input.lower()

        ticket_keywords = [
            "lỗi", "hỏng", "sự cố", "vấn đề", "khiếu nại", "hỗ trợ",
            "sửa chữa", "báo lỗi", "cần xử lý", "nghiêm trọng", "gấp", "khẩn cấp"
        ]
        catalog_keywords = [
            "muốn xem", "giá dưới", "tìm", "tư vấn", "có xe", "có gói",
            "bao nhiêu tiền", "giá bao nhiêu", "gói du lịch", "tour", "giá"
        ]

        needs_ticket = any(kw in text for kw in ticket_keywords)
        needs_catalog = (not needs_ticket) and any(kw in text for kw in catalog_keywords)

        return {"needs_catalog": needs_catalog, "needs_ticket": needs_ticket}

    # -------------------------------------------------------------------
    # Parameter Extraction Helpers
    # -------------------------------------------------------------------
    def _parse_catalog_params(self, user_input: str):
        text = user_input.lower()
        if "du lịch" in text or "vinpearl" in text or "tour" in text or "nghỉ dưỡng" in text or "resort" in text:
            category = "du_lich"
        else:
            category = "xe_dien"

        max_price = 999999999999
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*triệu", text)
        if match:
            value = float(match.group(1).replace(",", "."))
            max_price = int(value * 1_000_000)

        return category, max_price

    def _parse_ticket_params(self, user_input: str):
        name_match = re.search(r"[Tt]ôi tên\s+([^,\.]+)", user_input)
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

        text = user_input.lower()
        if any(kw in text for kw in ["nghiêm trọng", "khẩn cấp", "gấp"]):
            priority = "high"
        elif "không gấp" in text or "thấp" in text:
            priority = "low"
        else:
            priority = "medium"

        return customer_name, user_input, priority

    # -------------------------------------------------------------------
    # Answer Formatting
    # -------------------------------------------------------------------
    def _format_catalog_answer(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn."

        lines = [
            f"- {p['name']}: {p['price_vnd']:,} VNĐ".replace(",", ".")
            for p in results
        ]
        return "Dưới đây là các sản phẩm phù hợp:\n" + "\n".join(lines)

    def _format_ticket_answer(self, ticket: Dict[str, Any]) -> str:
        return (
            f"Đã tạo ticket hỗ trợ {ticket['ticket_id']} cho khách hàng "
            f"{ticket['customer_name']} với mức ưu tiên {ticket['priority']}. "
            f"Đội ngũ CSKH sẽ liên hệ sớm nhất."
        )

    def _faq_answer(self, user_input: str) -> str:
        text = user_input.lower()
        if "bảo hành" in text and "pin" in text:
            return "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km, tùy điều kiện nào đến trước."
        return "Cảm ơn câu hỏi của bạn. VinAssistant sẽ hỗ trợ thêm thông tin chi tiết về sản phẩm/dịch vụ Vingroup."

    def _synthesize(self, tool_results: Dict[str, Any]) -> str:
        parts = []
        if "catalog" in tool_results:
            parts.append(self._format_catalog_answer(tool_results["catalog"]))
        if "ticket" in tool_results:
            parts.append(self._format_ticket_answer(tool_results["ticket"]))
        return "\n\n".join(parts)

    # -------------------------------------------------------------------
    # Agent Loop
    # -------------------------------------------------------------------
    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intent(user_input)
        tool_results: Dict[str, Any] = {}
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            if intents["needs_catalog"] and "catalog" not in tool_results:
                category, max_price = self._parse_catalog_params(user_input)
                results = search_product_catalog(category=category, max_price=max_price)
                tool_results["catalog"] = results
                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "action": "search_product_catalog",
                    "args": {"category": category, "max_price": max_price},
                    "observation": results
                })
                if not intents["needs_ticket"]:
                    answer = self._format_catalog_answer(results)
                    self.trace.append({"step": "final_answer", "answer": answer})
                    return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}
                continue

            if intents["needs_ticket"] and "ticket" not in tool_results:
                customer_name, issue_description, priority = self._parse_ticket_params(user_input)
                ticket = submit_support_ticket(
                    customer_name=customer_name,
                    issue_description=issue_description,
                    priority=priority
                )
                tool_results["ticket"] = ticket
                self.trace.append({
                    "step": f"iteration_{iteration}",
                    "action": "submit_support_ticket",
                    "args": {"customer_name": customer_name, "priority": priority},
                    "observation": ticket
                })
                if not intents["needs_catalog"] or "catalog" in tool_results:
                    answer = self._format_ticket_answer(ticket)
                    self.trace.append({"step": "final_answer", "answer": answer})
                    return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}
                continue

            # Không còn tool nào cần gọi -> FAQ trực tiếp hoặc tổng hợp kết quả đã có
            if tool_results:
                answer = self._synthesize(tool_results)
            else:
                answer = self._faq_answer(user_input)
            self.trace.append({"step": "final_answer", "answer": answer})
            return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}

        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": iteration,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
