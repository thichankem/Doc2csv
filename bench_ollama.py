"""Continuous Ollama benchmark for chunk latency validation."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Iterable

from src.ollama_client import generate


def build_chunk(word_count: int = 200) -> str:
    sentence = (
        "Hệ thống này giúp xử lý tài liệu dài, chuẩn hóa nội dung, trích xuất thông tin quan trọng "
        "và tạo dữ liệu huấn luyện chất lượng cao cho mô hình ngôn ngữ." 
    )
    words = sentence.split()
    repeat = max(1, (word_count + len(words) - 1) // len(words))
    chunk = " ".join(words * repeat)
    return " ".join(chunk.split()[:word_count])


def benchmark_once(model: str, num_predict: int, temperature: float, num_ctx: int) -> float:
    prompt = f"Tóm tắt ngắn gọn: {build_chunk()}"
    start = time.perf_counter()
    output = generate(
        model=model,
        prompt=prompt,
        temperature=temperature,
        num_ctx=num_ctx,
        options={"num_predict": num_predict},
    )
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.2f}s | {len(output):>4} chars | {output[:90].replace(chr(10), ' ')}")
    return elapsed


def run_benchmark(repeat: int, continuous: bool, interval: float, model: str, num_predict: int, temperature: float, num_ctx: int) -> None:
    durations: list[float] = []
    iteration = 0

    while True:
        iteration += 1
        duration = benchmark_once(model, num_predict, temperature, num_ctx)
        durations.append(duration)

        if continuous:
            time.sleep(interval)
            continue

        if iteration >= repeat:
            break

    if durations:
        avg = statistics.fmean(durations)
        fastest = min(durations)
        slowest = max(durations)
        p95 = sorted(durations)[max(0, int(len(durations) * 0.95) - 1)]
        print(
            f"\nTổng: {len(durations)} lần | Trung bình: {avg:.2f}s | Fastest: {fastest:.2f}s | "
            f"Slowest: {slowest:.2f}s | P95: {p95:.2f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark tốc độ Ollama cho chunk 200 từ")
    parser.add_argument("--model", default="llama3.1:8b", help="Tên model Ollama")
    parser.add_argument("--repeat", type=int, default=5, help="Số lần benchmark hoặc 0 nếu chạy liên tục")
    parser.add_argument("--continuous", action="store_true", help="Chạy liên tục cho đến khi dừng bằng Ctrl+C")
    parser.add_argument("--interval", type=float, default=1.0, help="Khoảng nghỉ giữa các vòng khi chạy liên tục")
    parser.add_argument("--num-predict", type=int, default=64, help="Giới hạn token sinh")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature")
    parser.add_argument("--num-ctx", type=int, default=8192, help="num_ctx")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_benchmark(
            repeat=args.repeat,
            continuous=args.continuous,
            interval=args.interval,
            model=args.model,
            num_predict=args.num_predict,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
        )
    except KeyboardInterrupt:
        print("\nDừng benchmark.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
