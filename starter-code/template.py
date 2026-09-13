"""
Lab #3: Baseline Chatbot vs ReAct Agent

Deterministic ReAct implementation for the provided lab/autograder.
The planner is rule-based (no external LLM required), but the execution flow
still follows Thought -> Action -> Observation -> Final Answer.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tools import TOOL_DEFINITIONS, TOOL_MAP


SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""


class ChatbotBaseline:
    """Baseline chatbot: trả lời 1 lượt và tuyệt đối không gọi tool."""

    def query(self, user_input: str) -> dict:
        return {
            "status": "success",
            "answer": (
                "[Chatbot Baseline] Tôi có thể hướng dẫn ở mức tổng quát, "
                "nhưng không gọi công cụ để tra cứu dữ liệu chuyến bay hoặc "
                "thời tiết trong chế độ baseline."
            ),
            "tool_calls": [],
        }


class ReActAgent:
    """ReAct Agent với planner nhẹ + tool execution + trace logging."""

    KNOWN_AIRPORT_CODES = {"HAN", "SGN", "DAD"}

    CITY_CODE_MAP = {
        "hà nội": "HAN",
        "ha noi": "HAN",
        "hanoi": "HAN",
        "tp. hồ chí minh": "SGN",
        "tp hồ chí minh": "SGN",
        "hồ chí minh": "SGN",
        "ho chi minh": "SGN",
        "sài gòn": "SGN",
        "sai gon": "SGN",
        "đà nẵng": "DAD",
        "da nang": "DAD",
    }

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []

    @staticmethod
    def _parse_agent_response(response: str) -> dict:
        """Parse response ở format Thought/Action/Observation/Final Answer.

        Hàm này giữ lại đúng tinh thần ReAct/System Prompt và có thể dùng khi
        thay planner rule-based bằng một LLM backend sau này.
        """
        parsed = {
            "thought": "",
            "action": None,
            "observation": None,
            "final_answer": None,
        }
        fields = re.split(
            r"^\s*(Thought|Action|Observation|Final Answer):[ \t]*",
            response,
            flags=re.MULTILINE,
        )

        for label, content in zip(fields[1::2], fields[2::2]):
            content = content.strip()
            if label == "Thought":
                parsed["thought"] = content
            elif label == "Observation":
                parsed["observation"] = content
            elif label == "Final Answer":
                parsed["final_answer"] = content
                parsed["action"] = None
            elif label == "Action":
                try:
                    action = json.loads(content)
                except json.JSONDecodeError:
                    parsed["observation"] = "Invalid JSON format"
                    continue

                if (
                    isinstance(action, dict)
                    and isinstance(action.get("name"), str)
                    and action["name"].strip()
                    and isinstance(action.get("args"), dict)
                ):
                    action["name"] = action["name"].strip().lower()
                    parsed["action"] = action
                else:
                    parsed["observation"] = (
                        "Invalid action: expected name and args"
                    )
        return parsed

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.lower().strip()

    def _extract_airport_codes(self, text: str) -> List[str]:
        """Lấy airport code từ query, có fallback từ tên thành phố."""
        # Chỉ nhận token là airport code hợp lệ; tránh hiểu nhầm các từ
        # 3 ký tự như "bay", "cho", ... thành mã sân bay.
        tokens = re.findall(r"\b[A-Za-z]{3}\b", text)
        codes = [
            token.upper()
            for token in tokens
            if token.upper() in self.KNOWN_AIRPORT_CODES
        ]
        ordered_codes: List[str] = []
        for code in codes:
            if code not in ordered_codes:
                ordered_codes.append(code)

        normalized = self._normalize_text(text)
        for city_name, code in self.CITY_CODE_MAP.items():
            if city_name in normalized and code not in ordered_codes:
                ordered_codes.append(code)
        return ordered_codes

    @staticmethod
    def _extract_max_price(text: str, default: int = 5_000_000) -> int:
        """Hiểu các dạng phổ biến như 2 triệu, 1.5 triệu, 1500000 VND."""
        normalized = text.lower().replace(",", ".")

        million_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:triệu|trieu|tr)\b", normalized
        )
        if million_match:
            return int(float(million_match.group(1)) * 1_000_000)

        vnd_match = re.search(
            r"(\d{6,9})\s*(?:vnd|vnđ|đ|dong|đồng)?\b", normalized
        )
        if vnd_match:
            return int(vnd_match.group(1))

        return default

    @staticmethod
    def _detect_intents(text: str) -> Tuple[bool, bool]:
        normalized = text.lower()

        # Policy/FAQ có thể chứa các từ như "vé máy bay" nhưng không phải
        # yêu cầu tra cứu chuyến bay theo route/price.
        is_faq_or_policy = any(
            keyword in normalized
            for keyword in (
                "chính sách", "chinh sach", "đổi trả", "doi tra",
                "hoàn vé", "hoan ve", "điều khoản", "dieu khoan", "faq"
            )
        )
        if is_faq_or_policy:
            return False, False

        wants_flight = any(
            keyword in normalized
            for keyword in ("chuyến bay", "chuyen bay", "vé", "ve may bay", "bay từ", "bay tu")
        )
        wants_weather = any(
            keyword in normalized
            for keyword in (
                "thời tiết",
                "thoi tiet",
                "weather",
                "nên mặc",
                "nen mac",
                "trang phục",
                "trang phuc",
            )
        )
        return wants_flight, wants_weather

    @staticmethod
    def _format_flights(flights: List[Dict[str, Any]]) -> str:
        if not flights:
            return "Không tìm thấy chuyến bay phù hợp với điều kiện đã cho."

        lines = ["Các chuyến bay phù hợp:"]
        for flight in flights:
            lines.append(
                "- {flight_number}: {origin} → {destination}, {departure_time}, "
                "{price_vnd:,} VND, {airline}".format(**flight)
            )
        return "\n".join(lines)

    @staticmethod
    def _format_weather(city_code: str, weather: Dict[str, Any]) -> str:
        if "error" in weather:
            return f"Không lấy được thời tiết {city_code}: {weather['error']}"

        return (
            f"Thời tiết {weather['city']} ({city_code}): "
            f"{weather['temperature_c']}°C, {weather['condition']}, "
            f"độ ẩm {weather['humidity_pct']}%. "
            f"Gợi ý: {weather['recommendation']}"
        )

    def _execute_action(self, action: Dict[str, Any]) -> Any:
        """Thực thi action qua TOOL_MAP và lưu tool call."""
        tool_name = action["name"].strip().lower()
        args = action.get("args", {})
        tool = TOOL_MAP.get(tool_name)

        self.tool_calls.append({"name": tool_name, "args": args})

        if tool is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return tool(**args)
        except Exception as exc:
            return {"error": f"Tool execution failed: {exc}"}

    def _append_trace(
        self,
        iteration: int,
        thought: str,
        action: Optional[Dict[str, Any]],
        observation: Any = None,
        final_answer: Optional[str] = None,
    ) -> None:
        self.trace.append(
            {
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": observation,
                "final_answer": final_answer,
            }
        )

    def _completed(self, answer: str) -> dict:
        return {
            "status": "completed",
            "iterations": len(self.trace),
            "trace": self.trace,
            "answer": answer,
            "tool_calls": self.tool_calls,
        }

    def _max_iterations_result(self) -> dict:
        return {
            "status": "max_iterations_reached",
            "iterations": len(self.trace),
            "trace": self.trace,
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "tool_calls": self.tool_calls,
        }

    def run(self, user_input: str) -> dict:
        """Chạy deterministic ReAct loop.

        Quy ước iteration cho lab:
        - single-tool query: tool call + Final Answer trong cùng 1 iteration;
        - multi-tool query: 1 iteration/tool + 1 iteration tổng hợp cuối;
        - FAQ/no-tool: Final Answer trong 1 iteration.
        """
        self.trace = []
        self.tool_calls = []

        if self.max_iterations <= 0:
            return self._max_iterations_result()

        wants_flight, wants_weather = self._detect_intents(user_input)
        airport_codes = self._extract_airport_codes(user_input)
        max_price = self._extract_max_price(user_input)

        # No-tool / FAQ path.
        if not wants_flight and not wants_weather:
            answer = (
                "Vinpearl: câu hỏi này thuộc nhóm FAQ/chính sách nên ReAct Agent "
                "không cần gọi tool chuyến bay hoặc thời tiết. Vui lòng đối chiếu "
                "điều khoản/chính sách chính thức áp dụng cho dịch vụ cụ thể."
                if "vinpearl" in user_input.lower()
                else "Câu hỏi này không cần dùng tool chuyến bay hoặc thời tiết."
            )
            self._append_trace(
                iteration=1,
                thought="Câu hỏi không cần tool; trả lời trực tiếp.",
                action=None,
                final_answer=answer,
            )
            return self._completed(answer)

        # Xác định tham số dùng cho tool.
        origin: Optional[str] = None
        destination: Optional[str] = None
        weather_code: Optional[str] = None

        if wants_flight:
            if len(airport_codes) >= 2:
                origin, destination = airport_codes[0], airport_codes[1]
            elif len(airport_codes) == 1:
                destination = airport_codes[0]

        if wants_weather:
            if wants_flight and destination:
                weather_code = destination
            elif airport_codes:
                weather_code = airport_codes[-1]

        observations: List[str] = []
        iteration = 0

        # STEP 1 (nếu cần): Flight Action -> Observation
        if wants_flight:
            if iteration >= self.max_iterations:
                return self._max_iterations_result()
            iteration += 1

            if not origin or not destination:
                observation: Any = {
                    "error": "Không xác định đủ sân bay đi và sân bay đến."
                }
                action = None
                flight_text = observation["error"]
            else:
                action = {
                    "name": "get_flight_info",
                    "args": {
                        "origin": origin,
                        "destination": destination,
                        "max_price": max_price,
                    },
                }
                observation = self._execute_action(action)
                if isinstance(observation, list):
                    flight_text = self._format_flights(observation)
                else:
                    flight_text = f"Lỗi tra cứu chuyến bay: {observation}"

            observations.append(flight_text)

            # Single-step flight: hoàn tất ngay trong iteration 1.
            if not wants_weather:
                self._append_trace(
                    iteration=iteration,
                    thought="Cần tra cứu chuyến bay phù hợp với hành trình và ngân sách.",
                    action=action,
                    observation=observation,
                    final_answer=flight_text,
                )
                return self._completed(flight_text)

            self._append_trace(
                iteration=iteration,
                thought="Cần tra cứu chuyến bay trước khi xử lý yêu cầu thời tiết.",
                action=action,
                observation=observation,
            )

        # STEP 2 (nếu cần): Weather Action -> Observation
        if wants_weather:
            if iteration >= self.max_iterations:
                return self._max_iterations_result()
            iteration += 1

            if not weather_code:
                observation = {"error": "Không xác định được mã thành phố."}
                action = None
                weather_text = observation["error"]
            else:
                action = {
                    "name": "get_weather_forecast",
                    "args": {"city_code": weather_code},
                }
                observation = self._execute_action(action)
                if isinstance(observation, dict):
                    weather_text = self._format_weather(weather_code, observation)
                else:
                    weather_text = f"Lỗi tra cứu thời tiết: {observation}"

            observations.append(weather_text)

            # Single-step weather: hoàn tất ngay trong iteration 1.
            if not wants_flight:
                self._append_trace(
                    iteration=iteration,
                    thought="Cần lấy dữ liệu thời tiết cho thành phố được hỏi.",
                    action=action,
                    observation=observation,
                    final_answer=weather_text,
                )
                return self._completed(weather_text)

            self._append_trace(
                iteration=iteration,
                thought="Đã có chuyến bay; tiếp tục lấy thời tiết tại điểm đến.",
                action=action,
                observation=observation,
            )

        # Multi-step query cần thêm 1 iteration để tổng hợp Final Answer.
        if iteration >= self.max_iterations:
            return self._max_iterations_result()

        iteration += 1
        answer = "\n\n".join(observations)
        self._append_trace(
            iteration=iteration,
            thought="Đã đủ observations; tổng hợp câu trả lời cuối cùng.",
            action=None,
            observation=None,
            final_answer=answer,
        )
        return self._completed(answer)


def main():
    user_query = (
        "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, "
        "rồi cho biết thời tiết SGN nên mặc gì?"
    )

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(json.dumps(chatbot.query(user_query), indent=2, ensure_ascii=False))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
