#!/usr/bin/env python3
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = r'''
#include "runtimes/kokoro/src/internal/chunk.h"

#include <cassert>
#include <string>
#include <vector>

using chirp_kokoro::chunk_text;

int main() {
    {
        std::vector<std::string> chunks = chunk_text("Hello world!  How are you? Fine, thanks.");
        assert((chunks == std::vector<std::string>{"Hello world!", "How are you?", "Fine,", "thanks."}));
    }
    {
        std::vector<std::string> chunks = chunk_text("  Wait...  go  ");
        assert((chunks == std::vector<std::string>{"Wait.", ".", ".", "go"}));
    }
    {
        std::vector<std::string> chunks = chunk_text("No punctuation here");
        assert((chunks == std::vector<std::string>{"No punctuation here"}));
    }
    return 0;
}
'''
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        test_cpp = td_path / "chunk_test.cpp"
        exe = td_path / "chunk_test"
        test_cpp.write_text(source)
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-I",
                str(ROOT),
                str(test_cpp),
                str(ROOT / "runtimes/kokoro/src/internal/chunk.cpp"),
                "-o",
                str(exe),
            ],
            check=True,
        )
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
