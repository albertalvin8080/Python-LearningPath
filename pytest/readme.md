## Reminder
Add the root directory of source/ and tests/ inside the module search path (sys.path):
```
python -m tests.test_calculator
```
This allows imports like `import source.calc` to work because `source` is now in a directory that Python knows how to find.

## Running tests
In order to run the desired test, you need to follow the syntax:
```
pytest tests/<test_mytest>.py
```
Or, if you want to run all tests at once:
```
pytest tests/
```