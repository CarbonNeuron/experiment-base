# Replace tqdm with Rich progress bars

## Goal

Replace ALL `tqdm` usage with `rich.progress.Progress` for visual consistency with the existing `PrettyLogger` (in `training/logger.py`). The logger already uses `Console(stderr=True)` — the progress bars should share that same console.

## What to change

### 1. `training/logger.py` — add progress-bar factory methods

Add methods to `PrettyLogger` that return configured `rich.progress.Progress` contexts:

```python
def training_progress(self, total: int, initial: int = 0) -> Progress:
    """Return a Rich Progress context for the main training loop."""
    # Use self.console so output doesn't fight with other logger output
    # Columns: spinner, description, bar, percentage, step counter, elapsed, speed, ETA
    # Include a live-updating postfix area (like tqdm's set_postfix)

def validation_progress(self, total: int) -> Progress:
    """Return a Rich Progress context for validation."""
    # Simpler: spinner, description, bar, percentage, completed/total, elapsed

def tokenization_progress(self, total: int, description: str) -> Progress:
    """Return a Rich Progress context for data tokenization."""
    # Simple: description, bar, percentage, completed/total, speed, ETA
```

For the training progress bar, we need tqdm's `set_postfix` equivalent. Rich handles this via custom columns or by updating the task description. The cleanest approach:
- Add a `MofNCompleteColumn` or similar for "step X/Y"
- For the postfix metrics (epoch, loss, lr, hard, margin), use a custom `TextColumn` that reads from `task.fields` — Rich tasks support arbitrary fields that can be updated via `progress.update(task_id, field=value)`

When `_RICH_AVAILABLE` is False, these methods should return a context manager that wraps tqdm as fallback (so call sites don't need to branch).

### 2. `training/trainer.py` — replace tqdm in `fit()` and `evaluate()`

**`evaluate()` (lines 187-193):** Replace `tqdm(self.val_loader, ...)` with:
```python
with self.logger.validation_progress(progress_total) as progress:
    task = progress.add_task("Validating", total=progress_total)
    for chunk in self.val_loader:
        ...
        progress.advance(task)
        if max_batches > 0 and batches >= max_batches:
            break
```

**`fit()` (lines 242-249):** Replace the outer `tqdm(total=..., desc="Training", ...)` context with:
```python
with self.logger.training_progress(self.total_steps, initial=self.step) as progress:
    task = progress.add_task("Training", total=self.total_steps, completed=min(self.step, self.total_steps))
    ...
```

Replace `progress.set_postfix(...)` (line 310) with `progress.update(task, **postfix)` or equivalent field update.
Replace `progress.update(1)` (line 312) with `progress.advance(task)`.

Remove `progress.write()` calls if any remain (should already be replaced by logger calls from the previous pass).

Remove the `from tqdm.auto import tqdm` import from `training/trainer.py`.

### 3. `data.py` — replace tqdm in tokenization loop

**Line 53:** Replace `tqdm(raw["text"], desc=f"Tokenizing {split}", unit="doc")` with a Rich progress bar from the logger:
```python
with logger.tokenization_progress(len(raw["text"]), f"Tokenizing {split}") as prog:
    tok_task = prog.add_task(f"Tokenizing {split}", total=len(raw["text"]))
    for text in raw["text"]:
        ...
        prog.advance(tok_task)
```

Remove the `from tqdm.auto import tqdm` import from `data.py`.

### 4. Remove tqdm dependency

After all replacements, check if tqdm is imported ANYWHERE still. If not, it can be removed from `requirements.txt` (if listed). Don't remove it if other files still use it.

## Design constraints

- Progress bars MUST use the same `Console` instance as the logger (`self.console`) to avoid output corruption
- The training progress bar must show: current step, total steps, percentage, elapsed time, speed (steps/s), and the live metrics (epoch, loss, lr, and optionally hard/margin)
- `transient=True` on validation/tokenization progress bars (they disappear when done, keeping output clean)
- Training progress bar should NOT be transient (stays visible)
- Rich fallback: when `_RICH_AVAILABLE` is False, wrap tqdm in a compatible context manager so call sites stay clean

## Files to modify

- `training/logger.py` — add progress factory methods
- `training/trainer.py` — replace tqdm usage, remove tqdm import
- `data.py` — replace tqdm usage, remove tqdm import

## Files NOT to modify

- Everything else — same rules as before

## Verification

- `python -m pytest tests/ -v` — all tests must pass
- `python -c "from training.logger import PrettyLogger; p = PrettyLogger(); print('ok')"` — import works
- Grep for remaining `tqdm` imports: `grep -rn "tqdm" --include="*.py"` — should only appear in files outside the modified set (scripts/, benchmarks/, etc.) or not at all
