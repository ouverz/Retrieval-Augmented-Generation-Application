# CLAUDE.md

This file provides guidanc to Claude Code (claude.ai/code) when working with
code in the repository.

# Core Philosophy:
- "Should work" != "Does work" - patter matching isn't enough
- I'm not paid to write code, rather to solve problems
- Untested code is just a guess, not a solution
- Test changes instead of assuming they work
- Verify output match expecatations
- Handle error properly
- Follow the complete checklist

# Coding Standards
- Python codes: follow PEP8 rules and guidelines
- max length of lines is 80 characters
- ensure to always create a pydantic schema for any data model
- ensure all functions have descriptive namings (ex: functionXYZ() is 
a name. check_seed_d() and is_valid_product() are better)
- ensure all functions have a short 1-line comment that describes what
the function does.


## Development Commands

- **Activate virtual environment**: `source venv/bin/activate` if using venv
- **Install dependencies**: `uv sync` (preferred) or `pip install -r requirements.txt`
- **Setup NLTK data**: `python scripts/setup_nltk.py` (run once after dependencies)


### Testing
- **Run tests**: `pytest`
- **Run tests with coverage**: `pytest -cov=app`
- **Run specific test file**: `pytest tests/test-xxx/test_xxx_zzz.py`
