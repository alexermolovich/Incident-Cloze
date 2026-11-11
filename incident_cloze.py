"""Compare compact masked language models on incident-response dialogue."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TaskRow = dict[str, str]
RetainedView = tuple[tuple[int, ...], int]
RetainedRecordViews = tuple[RetainedView, ...]

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data/incident_dialogues.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/incident_cloze_results.xlsx"
RECORDS_SHEET_NAME = "Records"
ALL_COMBINATIONS_SHEET_NAME = "All_combinations"
MASK_LITERAL = "[MASK]"
LAYERS_VALUES = (2, 4)
MAX_INPUT_TOKEN_VALUES = tuple(range(32, 97))
CONTEXT_VIEW_VALUES = (1, 2, 3, 4)
TEMPERATURE_VALUES = tuple(value / 100 for value in range(50, 201, 5))
REQUIRED_COLUMNS = (
    "Record_ID",
    "Conversation_ID",
    "Incident_Type",
    "Speaker_Role",
    "Conversation",
    "Masked_Utterance",
    "Correct_Token",
)
OUTPUT_COLUMNS = (
    "Layers",
    "Max_input_tokens",
    "Context_views",
    "Temperature",
    "Accuracy",
)
BATCH_SIZE = 64


@dataclass(frozen=True)
class EvaluationGrid:
    layers: tuple[int, ...] = LAYERS_VALUES
    token_limits: tuple[int, ...] = MAX_INPUT_TOKEN_VALUES
    view_counts: tuple[int, ...] = CONTEXT_VIEW_VALUES
    temperatures: tuple[float, ...] = TEMPERATURE_VALUES

    @property
    def configuration_count(self) -> int:
        return len(self.layers) * len(self.token_limits) * len(self.view_counts) * len(self.temperatures)


def _is_blank(value: str) -> bool:
    return value.strip() == ""


def _normalize_task_row(raw_task: dict[str, Any]) -> TaskRow:
    task = {
        column_name: "" if raw_task[column_name] is None else str(raw_task[column_name])
        for column_name in REQUIRED_COLUMNS
    }
    task["Conversation"] = task["Conversation"].replace("\\n", "\n")
    return task


def _build_context_view(
    earlier_lines: list[str],
    final_masked_line: str,
    line_count: int | None,
) -> str:
    if line_count == 0:
        return final_masked_line

    selected_lines = earlier_lines if line_count is None else earlier_lines[-line_count:]
    return "\n".join([*selected_lines, final_masked_line])


def _open_input_workbook(workbook_path: Path) -> pd.ExcelFile:
    try:
        return pd.ExcelFile(workbook_path)
    except ImportError as exc:
        raise RuntimeError(
            "Reading workbooks requires the 'pandas' package and the 'openpyxl' engine."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Workbook is not a readable .xlsx file: {workbook_path}") from exc


def _read_records_sheet(workbook: pd.ExcelFile, workbook_path: Path) -> pd.DataFrame:
    try:
        records_dataframe = pd.read_excel(
            workbook,
            sheet_name=RECORDS_SHEET_NAME,
            dtype=str,
            keep_default_na=False,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Reading workbooks requires the 'pandas' package and the 'openpyxl' engine."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Workbook is not a readable .xlsx file: {workbook_path}") from exc

    if len(records_dataframe.columns) == 0:
        raise ValueError(f"{RECORDS_SHEET_NAME} is empty.")

    records_dataframe.columns = [str(column_name) for column_name in records_dataframe.columns]
    return records_dataframe


def read_restoration_tasks(dataset_path: Path) -> list[TaskRow]:
    if not dataset_path.is_file():
        raise ValueError(f"Dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".csv":
        records_dataframe = pd.read_csv(dataset_path, dtype=str, keep_default_na=False)
    elif dataset_path.suffix.lower() == ".xlsx":
        with _open_input_workbook(dataset_path) as workbook:
            records_dataframe = _read_records_sheet(workbook, dataset_path)
    else:
        raise ValueError("The dataset must use the .csv or .xlsx extension.")

    missing_columns = [
        column_name for column_name in REQUIRED_COLUMNS if column_name not in records_dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{RECORDS_SHEET_NAME} is missing required column(s): {', '.join(missing_columns)}"
        )

    records_dataframe = records_dataframe.loc[:, list(REQUIRED_COLUMNS)]
    tasks: list[TaskRow] = []
    seen_ids: set[str] = set()
    for row_number, raw_task in enumerate(records_dataframe.to_dict(orient="records"), start=2):
        task = _normalize_task_row(raw_task)
        if all(_is_blank(value) for value in task.values()):
            continue

        blank_columns = [column_name for column_name, value in task.items() if _is_blank(value)]
        if blank_columns:
            raise ValueError(
                f"Row {row_number} is missing required value(s): {', '.join(blank_columns)}"
            )

        if task["Record_ID"] in seen_ids:
            raise ValueError(f"Duplicate Record_ID: {task['Record_ID']}")
        seen_ids.add(task["Record_ID"])
        tasks.append(task)

    if not tasks:
        raise ValueError(f"{RECORDS_SHEET_NAME} does not contain any populated restoration tasks.")

    return tasks


def load_model_pairs() -> dict[int, dict[str, Any]]:
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model loading requires the 'transformers' package and a PyTorch backend such as 'torch'."
        ) from exc

    # Compare checkpoints with the same hidden size and different encoder depths.
    checkpoints = {
        2: "google/bert_uncased_L-2_H-128_A-2",
        4: "google/bert_uncased_L-4_H-128_A-2",
    }
    loaded_models: dict[int, dict[str, Any]] = {}

    try:
        for layers, checkpoint in checkpoints.items():
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForMaskedLM.from_pretrained(checkpoint)
            model.eval()
            loaded_models[layers] = {
                "tokenizer": tokenizer,
                "model": model,
            }
    except OSError as exc:
        raise RuntimeError(
            "Unable to load one or both checkpoints. Ensure the models are cached locally or "
            "the runtime can download them from Hugging Face."
        ) from exc

    return loaded_models


def build_context_windows(tasks: list[TaskRow]) -> list[tuple[str, str, str, str]]:
    context_views: list[tuple[str, str, str, str]] = []

    for task in tasks:
        if MASK_LITERAL in task["Conversation"]:
            raise ValueError(
                f"Record {task['Record_ID']} has {MASK_LITERAL} in Conversation, but it belongs only in "
                "Masked_Utterance."
            )

        final_masked_line = f"{task['Speaker_Role']}: {task['Masked_Utterance']}"
        if final_masked_line.count(MASK_LITERAL) != 1:
            raise ValueError(
                f"Record {task['Record_ID']} must contain exactly one {MASK_LITERAL} token in the final "
                "speaker-prefixed masked utterance."
            )

        earlier_lines = task["Conversation"].splitlines()
        context_views.append(
            (
                _build_context_view(earlier_lines, final_masked_line, 0),
                _build_context_view(earlier_lines, final_masked_line, 2),
                _build_context_view(earlier_lines, final_masked_line, 4),
                _build_context_view(earlier_lines, final_masked_line, None),
            )
        )

    return context_views


def prepare_model_inputs(
    tasks: list[TaskRow],
    context_views: list[tuple[str, str, str, str]],
    loaded_models: dict[int, dict[str, Any]],
    grid: EvaluationGrid = EvaluationGrid(),
) -> dict[int, dict[str, Any]]:
    prepared_inputs: dict[int, dict[str, Any]] = {}

    for layers in grid.layers:
        tokenizer = loaded_models[layers]["tokenizer"]
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        mask_token_id = tokenizer.mask_token_id
        pad_token_id = tokenizer.pad_token_id
        if cls_token_id is None or sep_token_id is None or mask_token_id is None or pad_token_id is None:
            raise ValueError("Tokenizer must define [CLS], [SEP], [MASK], and padding token IDs.")

        tokenized_views: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = [
            tuple(
                tuple(tokenizer.encode(view_text, add_special_tokens=False))
                for view_text in record_views
            )
            for record_views in context_views
        ]

        correct_token_ids: list[int] = []
        for task in tasks:
            correct_id = tokenizer.convert_tokens_to_ids(task["Correct_Token"])
            if correct_id is None or correct_id == tokenizer.unk_token_id:
                raise ValueError(
                    f"Record {task['Record_ID']} has Correct_Token {task['Correct_Token']!r}, which is not "
                    "a known vocabulary token."
                )
            correct_token_ids.append(int(correct_id))

        retained_views: dict[int, list[RetainedRecordViews]] = {}
        for max_input_tokens in grid.token_limits:
            views_for_length: list[RetainedRecordViews] = []
            for task, record_views in zip(tasks, tokenized_views):
                retained_record_views: list[RetainedView] = []
                for view_number, token_ids in enumerate(record_views, start=1):
                    retained_token_ids = token_ids[-(max_input_tokens - 2):]
                    input_ids = (cls_token_id, *retained_token_ids, sep_token_id)
                    mask_count = input_ids.count(mask_token_id)
                    if mask_count != 1:
                        raise ValueError(
                            f"Record {task['Record_ID']} View {view_number} does not retain exactly one "
                            f"mask token at Max_input_tokens={max_input_tokens}."
                        )
                    retained_record_views.append((input_ids, input_ids.index(mask_token_id)))
                views_for_length.append(tuple(retained_record_views))
            retained_views[max_input_tokens] = views_for_length

        prepared_inputs[layers] = {
            "correct_token_ids": np.asarray(correct_token_ids, dtype=np.int64),
            "pad_token_id": int(pad_token_id),
            "retained_views": retained_views,
        }

    return prepared_inputs


def _require_torch(step_label: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"{step_label} requires the 'torch' package to run model inference.") from exc

    return torch


def _collect_view_logits(
    model: Any,
    records_for_length: list[RetainedRecordViews],
    pad_token_id: int,
    torch: Any,
) -> np.ndarray:
    unique_sequences: dict[tuple[int, ...], int] = {}
    for record_views in records_for_length:
        for input_ids, mask_index in record_views:
            previous_mask_index = unique_sequences.setdefault(input_ids, mask_index)
            if previous_mask_index != mask_index:
                raise ValueError("The same retained sequence produced different mask positions.")

    logits_cache: dict[tuple[int, ...], np.ndarray] = {}
    unique_items = list(unique_sequences.items())
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_start in range(0, len(unique_items), BATCH_SIZE):
            batch_items = unique_items[batch_start : batch_start + BATCH_SIZE]
            batch_length = max(len(input_ids) for input_ids, _ in batch_items)
            batch_input_ids: list[list[int]] = []
            batch_attention_masks: list[list[int]] = []

            for input_ids, _ in batch_items:
                padding_length = batch_length - len(input_ids)
                # Right padding and attention masks preserve the retained text during batched inference.
                batch_input_ids.append(list(input_ids) + [pad_token_id] * padding_length)
                batch_attention_masks.append([1] * len(input_ids) + [0] * padding_length)

            model_output = model(
                input_ids=torch.tensor(batch_input_ids, dtype=torch.long, device=device),
                attention_mask=torch.tensor(
                    batch_attention_masks,
                    dtype=torch.long,
                    device=device,
                ),
            )
            batch_logits = model_output.logits.detach().cpu().to(torch.float32)

            for row_index, (input_ids, mask_index) in enumerate(batch_items):
                logits_cache[input_ids] = batch_logits[row_index, mask_index].numpy().copy()

    # Identical retained views still count in the average, but their logits only need one forward pass.
    view_logits = np.empty(
        (len(records_for_length), len(CONTEXT_VIEW_VALUES), int(model.config.vocab_size)),
        dtype=np.float32,
    )
    for record_index, record_views in enumerate(records_for_length):
        for view_index, (input_ids, _) in enumerate(record_views):
            view_logits[record_index, view_index] = logits_cache[input_ids]

    return view_logits


def _calculate_average_correct_token_probabilities(
    averaged_logits: np.ndarray,
    correct_token_ids: np.ndarray,
    temperatures: tuple[float, ...] = TEMPERATURE_VALUES,
) -> np.ndarray:
    temperature_array = np.asarray(temperatures, dtype=np.float64)
    average_probabilities = np.zeros(len(temperatures), dtype=np.float64)

    for record_index, correct_token_id in enumerate(correct_token_ids):
        shifted_logits = averaged_logits[record_index].astype(np.float64, copy=True)
        shifted_logits -= float(np.max(shifted_logits))
        # Positive temperature rescales softmax probabilities but leaves argmax and rank unchanged.
        scaled_logits = shifted_logits[:, np.newaxis] / temperature_array[np.newaxis, :]
        np.exp(scaled_logits, out=scaled_logits)
        average_probabilities += scaled_logits[int(correct_token_id)] / scaled_logits.sum(axis=0)

    return average_probabilities / len(correct_token_ids)


def _winner_key(result: dict[str, Any]) -> tuple[int, float, float]:
    return (
        int(result["Correct_tokens"]),
        float(result["Average_correct_token_probability"]),
        -float(result["Average_correct_token_rank"]),
    )


def score_context_ensembles(
    prepared_inputs: dict[int, dict[str, Any]],
    loaded_models: dict[int, dict[str, Any]],
    grid: EvaluationGrid = EvaluationGrid(),
) -> list[dict[str, Any]]:
    torch = _require_torch("Context scoring")

    base_results: list[dict[str, Any]] = []

    for layers in grid.layers:
        layer_inputs = prepared_inputs[layers]
        model = loaded_models[layers]["model"]
        correct_token_ids = layer_inputs["correct_token_ids"]
        pad_token_id = layer_inputs["pad_token_id"]
        retained_views = layer_inputs["retained_views"]
        record_indexes = np.arange(len(correct_token_ids))

        for max_input_tokens in grid.token_limits:
            records_for_length = retained_views[max_input_tokens]
            view_logits = _collect_view_logits(model, records_for_length, pad_token_id, torch)
            cumulative_logits = np.cumsum(view_logits, axis=1, dtype=np.float32)

            for context_views in grid.view_counts:
                averaged_logits = cumulative_logits[:, context_views - 1, :] / np.float32(context_views)
                predicted_token_ids = np.argmax(averaged_logits, axis=1)
                correct_tokens = int(np.count_nonzero(predicted_token_ids == correct_token_ids))
                correct_logits = averaged_logits[record_indexes, correct_token_ids]
                correct_token_ranks = (
                    np.count_nonzero(averaged_logits > correct_logits[:, np.newaxis], axis=1) + 1
                )

                base_results.append(
                    {
                        "Layers": layers,
                        "Max_input_tokens": max_input_tokens,
                        "Context_views": context_views,
                        "Correct_tokens": correct_tokens,
                        "Accuracy": float(correct_tokens / len(correct_token_ids)),
                        "Average_correct_token_rank": float(
                            np.mean(correct_token_ranks, dtype=np.float64)
                        ),
                    }
                )

    return base_results


def expand_temperature_grid(
    base_results: list[dict[str, Any]],
    grid: EvaluationGrid = EvaluationGrid(),
) -> list[dict[str, Any]]:
    configuration_results = [
        base_result | {"Temperature": temperature}
        for base_result in base_results
        for temperature in grid.temperatures
    ]

    # Temperature must still appear in the output workbook even though it does not change accuracy or rank.
    if len(configuration_results) != grid.configuration_count:
        raise RuntimeError(
            f"The configuration search did not produce all {grid.configuration_count:,} combinations."
        )

    return configuration_results


def select_best_configuration(
    prepared_inputs: dict[int, dict[str, Any]],
    loaded_models: dict[int, dict[str, Any]],
    base_results: list[dict[str, Any]],
    grid: EvaluationGrid = EvaluationGrid(),
) -> dict[str, Any]:
    torch = _require_torch("Configuration selection")

    best_correct_tokens = max(base_result["Correct_tokens"] for base_result in base_results)
    candidate_results = [
        base_result
        for base_result in base_results
        if base_result["Correct_tokens"] == best_correct_tokens
    ]
    grouped_candidates: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for candidate_result in candidate_results:
        key = (
            candidate_result["Layers"],
            candidate_result["Max_input_tokens"],
        )
        grouped_candidates.setdefault(key, []).append(candidate_result)

    best_result: dict[str, Any] | None = None

    for (layers, max_input_tokens), results_for_pair in grouped_candidates.items():
        layer_inputs = prepared_inputs[layers]
        model = loaded_models[layers]["model"]
        correct_token_ids = layer_inputs["correct_token_ids"]
        pad_token_id = layer_inputs["pad_token_id"]
        records_for_length = layer_inputs["retained_views"][max_input_tokens]
        view_logits = _collect_view_logits(model, records_for_length, pad_token_id, torch)
        cumulative_logits = np.cumsum(view_logits, axis=1, dtype=np.float32)

        for candidate_result in results_for_pair:
            averaged_logits = (
                cumulative_logits[:, candidate_result["Context_views"] - 1, :]
                / np.float32(candidate_result["Context_views"])
            )
            average_probabilities = _calculate_average_correct_token_probabilities(
                averaged_logits,
                correct_token_ids,
                grid.temperatures,
            )
            best_temperature_index = int(np.argmax(average_probabilities))
            current_result = candidate_result | {
                "Temperature": grid.temperatures[best_temperature_index],
                "Average_correct_token_probability": float(
                    average_probabilities[best_temperature_index]
                ),
            }

            if best_result is None or _winner_key(current_result) > _winner_key(best_result):
                best_result = current_result

    if best_result is None:
        raise RuntimeError("The winning configuration could not be determined.")

    return best_result


def export_evaluation(
    configuration_results: list[dict[str, Any]],
    best_result: dict[str, Any],
    output_workbook_path: Path,
    run_metadata: dict[str, Any] | None = None,
) -> None:
    configuration_dataframe = pd.DataFrame(configuration_results, columns=OUTPUT_COLUMNS)

    try:
        output_workbook_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_workbook_path, engine="openpyxl") as writer:
            configuration_dataframe.to_excel(
                writer,
                sheet_name=ALL_COMBINATIONS_SHEET_NAME,
                index=False,
            )
            pd.DataFrame([best_result]).to_excel(writer, sheet_name="Best_configuration", index=False)
            if run_metadata:
                pd.DataFrame(run_metadata.items(), columns=["Setting", "Value"]).to_excel(
                    writer, sheet_name="Run_details", index=False,
                )
            worksheet = writer.book[ALL_COMBINATIONS_SHEET_NAME]
            # Accuracy is stored as a 0-1 fraction and displayed with Excel's percentage format.
            for accuracy_cell in worksheet["E"][1:]:
                accuracy_cell.number_format = "0.00%"
    except ImportError as exc:
        raise RuntimeError(
            "Exporting workbooks requires the 'pandas' package and the 'openpyxl' engine."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Unable to write output workbook: {output_workbook_path}") from exc

    print(f"Highest accuracy: {best_result['Accuracy']:.2%}")
    print(f"Layers: {best_result['Layers']}")
    print(f"Max_input_tokens: {best_result['Max_input_tokens']}")
    print(f"Context_views: {best_result['Context_views']}")
    print(f"Temperature: {best_result['Temperature']:.2f}")
    print(f"Results: {output_workbook_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Input .csv or .xlsx")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Results workbook (.xlsx)")
    parser.add_argument("--validate-only", action="store_true", help="Check records without loading models")
    parser.add_argument("--quick", action="store_true", help="Compare 32 configurations at a 64-token limit")
    args = parser.parse_args()
    grid = EvaluationGrid(token_limits=(64,), temperatures=(0.5, 1.0, 1.5, 2.0)) if args.quick else EvaluationGrid()

    try:
        dataset_path = args.dataset.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        if output_path.suffix.lower() != ".xlsx":
            raise ValueError("The output workbook must use the .xlsx extension.")
        if output_path == dataset_path:
            raise ValueError("The output workbook must differ from the input dataset.")
        tasks = read_restoration_tasks(dataset_path)
        context_views = build_context_windows(tasks)
        conversation_count = len({task["Conversation_ID"] for task in tasks})
        print(f"Incident Cloze: {len(tasks)} records across {conversation_count} incident threads.", flush=True)
        if args.validate_only:
            print("Dataset validation passed; vocabulary compatibility is checked when models load.")
            return

        print(f"Evaluating {grid.configuration_count:,} configurations.", flush=True)
        loaded_models = load_model_pairs()
        prepared_inputs = prepare_model_inputs(tasks, context_views, loaded_models, grid)
        base_results = score_context_ensembles(prepared_inputs, loaded_models, grid)
        configuration_results = expand_temperature_grid(base_results, grid)
        best_result = select_best_configuration(prepared_inputs, loaded_models, base_results, grid)
        export_evaluation(configuration_results, best_result, output_path, {
            "Dataset": str(dataset_path),
            "Dataset SHA256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "Records": len(tasks),
            "Incident threads": conversation_count,
            "Configurations": grid.configuration_count,
            "Search mode": "quick" if args.quick else "full",
            "Evaluation scope": "Configuration search on all records; no held-out test set",
        })
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
