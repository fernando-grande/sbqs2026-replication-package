"""Extract PDF text and ground the spreadsheet excerpts in it.

    python run_extract.py                 # anchor the spreadsheet excerpts only
    python run_extract.py --llm           # ground every excerpt with Gemini
    python run_extract.py --llm --limit 3 # try the LLM on the first 3 studies
"""
from extract.run import main

if __name__ == "__main__":
    main()
