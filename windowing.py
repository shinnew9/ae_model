"""
Windowing Utility
=================

This module contains the windowing utility for the audio files.
From `.wav` & `.tsv` files, it creates windowed `.wav` files with labels & ratios information.
"""

from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as F
import torchaudio  # type: ignore


class WindowingDict(TypedDict):
    window_st: int
    window_en: int
    iv_name: list[str]
    label_name: list[str]
    relative_ratio: list[float]
    absolute_ratio: list[float]


class Windowing:

    sr: int = 16000

    def __init__(
        self,
        audio_folders: str | Path | list[str | Path],
        similar_labels: dict[str, list[str]],
        window_save_folder: str | Path,
        window_size: float = 2.0,
        hop_size: float = 1.0,
        start_offset: float = 0.0,
        relative_ratio_threshold: float = 0.3,
        absolute_ratio_threshold: float = 0.9,
    ) -> None:

        self.audio_folders: list[Path]
        self.similar_labels: dict[str, list[str]] = similar_labels
        self.window_save_folder: str | Path = window_save_folder
        self.window_size: float = window_size
        self.hop_size: float = hop_size
        self.start_offset: float = start_offset
        self.relative_ratio_threshold: float = relative_ratio_threshold
        self.absolute_ratio_threshold: float = absolute_ratio_threshold

        if isinstance(audio_folders, (str, Path)):
            self.audio_folders = [Path(audio_folders)]
        else:
            self.audio_folders = [Path(folder) for folder in audio_folders]

    def __gather_audio_files(
        self,
    ) -> list[Path]:
        """Gather audio files in the folder."""

        audio_files: list[Path] = []
        for audio_folder in self.audio_folders:
            audio_files += list(audio_folder.rglob("*.wav"))

        return audio_files

    def __windowing(
        self,
        audio_filepath: str | Path,
        label_filepath: str | Path,
        save_path: str | Path,
    ) -> Path:
        """
        Window the long audio file and save as wav files in the folder.

        Arguments:
        ----------
        audio_filepath: str | Path
            Path to the audio file.
        label_filepath: str | Path
            Path to the label file.
            In sample point unit.

        Variables:
        ----------
        windowed_results: dict[int, dict[str, int | list[str] | list[float]]]
            {
                0: {
                    "window_st": 0,
                    "window_en": 16000,
                    "iv_name": ["breathing_heavily", ...],
                    "label_name": ["breathing_heavily", ...],
                    "relative_ratio": [1.0, ...],
                    "absolute_ratio": [1.0, ...],
                }, ...
            }
        """

        audio, sr = torchaudio.load(uri=audio_filepath)
        audio = audio.squeeze()

        assert audio.dim() == 1, "Audio must be mono."
        assert sr == self.sr, f"Sample rate must be {self.sr}."

        with open(file=label_filepath, mode="r", encoding="utf-8") as f:
            labels = f.read().strip().split("\n")
            f.close()
            print(f"Labels aread from {label_filepath}:{labels}")  # DEBUG

        audio_length: int = audio.shape[0]

        window_size: int = int(self.window_size * sr)
        hop_size: int = int(self.hop_size * sr)
        start_offset: int = int(self.start_offset * sr)

        total_length: int = max(audio_length, window_size)
        num_windows: int = (total_length - window_size) // hop_size + 1

        similar_labels = self.similar_labels

        # If the audio is slightly longer than the window size,
        # and the target audio is shorter than the window size,
        # place the target audio in the middle and make the window.
        windowed_results: dict[int, WindowingDict]
        st_int: int
        en_int: int

        if len(labels) == 1 and audio_length > window_size:
            st, en, label_name = labels[0].split("\t")
            st_int = int(st)
            en_int = int(en)
            # st_int = int(float(st)*sr)  # 시작시간 (float) -> (int)로 변환
            # en_int = int(float(en)*sr)  # 시작시간 (float) -> (int)로 변환


            assert (
                st_int < en_int
            ), f"Start time must be smaller than end time: \n\t - Line 1, {label_name}.\n"

            if en_int - st_int < window_size:
                found = False
                for iv_name, similars in similar_labels.items():
                    if label_name in similars:
                        found = True

                if not found:
                    print(f" -- Considering\n{label_name}\nas others.\n")
                    label_name = "others"

                windowed_results = {
                    0: {
                        "window_st": (en_int + st_int) // 2 - window_size // 2,
                        "window_en": (en_int + st_int) // 2 - window_size // 2 + window_size,
                        "iv_name": [label_name],
                        "label_name": [label_name],
                        "relative_ratio": [1.0],
                        "absolute_ratio": [1.0],
                    }
                }

        # If the audio is shorter than the window size,
        # pad the audio to the window size.
        if audio_length < window_size:
            margin = window_size - audio_length
            audio = F.pad(
                input=audio,
                pad=(margin // 2, margin - margin // 2),
            )
            audio_length = audio.shape[0]

        # To match the number of windows when start_offset > 0
        if start_offset > 0:
            audio = F.pad(
                input=audio,
                pad=(0, start_offset),
            )

        window_st: int
        window_en: int

        windowed_results = {}
        for i in range(num_windows):
            window_st = start_offset + i * hop_size
            window_en = window_st + window_size

            windowed_results[i] = {
                "window_st": window_st,
                "window_en": window_en,
                "iv_name": [],
                "label_name": [],
                "relative_ratio": [],
                "absolute_ratio": [],
            }

            oov_list: list[str] = []
            for j, label in enumerate(labels):
                st, en, label_name = label.split("\t")
                st_int = int(st)
                en_int = int(en)

                assert (
                    st_int < en_int
                ), f"Start time must be smaller than end time: \n\t - Line {j+1}, {label}.\n"

                if (
                    st_int < window_en and en_int > window_st
                ):  # Target overlapped with the window
                    # Calculate ratios
                    overlap: float = min(en_int, window_en) - max(st_int, window_st)
                    relative_ratio: float = overlap / window_size
                    absolute_ratio: float = overlap / (en_int - st_int)

                    found = False
                    for iv_name, similars in similar_labels.items():
                        if label_name in similars and (
                            relative_ratio > self.relative_ratio_threshold
                            or absolute_ratio > self.absolute_ratio_threshold
                        ):
                            windowed_results[i]["iv_name"].append(iv_name)
                            windowed_results[i]["label_name"].append(label_name)
                            windowed_results[i]["relative_ratio"].append(relative_ratio)
                            windowed_results[i]["absolute_ratio"].append(absolute_ratio)
                            found = True

                    if not found and label_name not in oov_list:
                        oov_list.append(label_name)

                else:  # Target not overlapped with the window
                    windowed_results[i]["iv_name"].append("others")
                    windowed_results[i]["label_name"].append("others")
                    windowed_results[i]["relative_ratio"].append(0.0)
                    windowed_results[i]["absolute_ratio"].append(0.0)
                    found = True

            if not found:
                windowed_results[i]["iv_name"].append("others")
                windowed_results[i]["label_name"].append("others")
                windowed_results[i]["relative_ratio"].append(relative_ratio)
                windowed_results[i]["absolute_ratio"].append(absolute_ratio)

        # `windowed_results` is set.
        # Now, save the windowed audio files.
        audio_filepath = Path(audio_filepath)
        filename: str = audio_filepath.stem
        (save_folder := Path(save_path) / filename).mkdir(parents=True, exist_ok=True)

        for window_idx, windowed_result in windowed_results.items():
            window_st = windowed_result["window_st"]
            window_en = windowed_result["window_en"]
            windowed_audio: torch.Tensor = audio[window_st:window_en]

            label_rel_abs_list: list[str] = []
            for iv_name, relative_ratio, absolute_ratio in zip(
                windowed_result["iv_name"],
                windowed_result["relative_ratio"],
                windowed_result["absolute_ratio"],
            ):
                label_rel_abs_list.append(
                    f"{iv_name.replace('_', '-')}-r{relative_ratio:.2f}-a{absolute_ratio:.2f}"
                )
            save_filename: str = f"{window_idx}_{'_'.join(label_rel_abs_list)}.wav"

            torchaudio.save(
                uri=save_folder / save_filename,
                src=windowed_audio.unsqueeze(dim=0),
                sample_rate=sr,
            )

        return save_folder

    def __call__(
        self,
    ) -> str | Path:

        audio_files: list[Path] = self.__gather_audio_files()

        for audio_file in audio_files:
            if audio_file.with_suffix(".tsv").exists():
                label_file = audio_file.with_suffix(".tsv")
            elif audio_file.with_suffix(".txt").exists():
                label_file = audio_file.with_suffix(".txt")

            self.__windowing(
                audio_filepath=audio_file,
                label_filepath=label_file,
                save_path=self.window_save_folder,
            )

        return self.window_save_folder
