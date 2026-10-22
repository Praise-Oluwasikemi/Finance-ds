"""Notebook-friendly grader: check(your_function) runs the same tests as pytest.

    from exercises.check import check
    check(compound_interest)
"""
import inspect
import traceback
from types import SimpleNamespace

import tests.test_exercises as suite


def check(func) -> bool:
    name = func.__name__
    test = getattr(suite, f"test_{name}", None)
    if test is None:
        print(f"No test found for '{name}'. Did you keep the original function name?")
        return False

    original = suite.ex
    suite.ex = SimpleNamespace(**{name: func})
    try:
        kwargs = {"raw_tx": suite.make_raw_tx()} if "raw_tx" in inspect.signature(test).parameters else {}
        test(**kwargs)
    except NotImplementedError:
        print(f"⏳ {name}: not started yet. Replace `raise NotImplementedError` with your code.")
        return False
    except AssertionError as e:
        line = traceback.extract_tb(e.__traceback__)[-1].line
        print(f"❌ {name}: a check failed.\n   Failing check: {line}\n   {e}".rstrip())
        return False
    except Exception as e:  # noqa: BLE001 - show any bug in the learner's code plainly
        print(f"💥 {name} raised {type(e).__name__}: {e}")
        return False
    finally:
        suite.ex = original
    print(f"✅ {name}: all checks passed!")
    return True
