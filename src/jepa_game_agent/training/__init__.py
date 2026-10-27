"""Train the world model in resumable chunks (Colab sessions end)."""

from jepa_game_agent.training.trainer import load_checkpoint, save_checkpoint, train

__all__ = ["load_checkpoint", "save_checkpoint", "train"]
