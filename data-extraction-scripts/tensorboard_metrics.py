from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import pandas as pd

class TensorBoardMetrics:
    def __init__(self, events_path:str, train_csv_path:str):
        """
        Args:
            events_path (str): path to tensorboard event file or directory
            train_csv_path (str): CSV containing training data
        """
        self.events_path = Path(events_path)
        self.train_csv_path = Path(train_csv_path)

        self.train_df = pd.read_csv(self.train_csv_path)

        if self.events_path.is_dir():
            event_files = sorted(self.events_path.glob("events.*"))
            if not event_files:
                raise FileNotFoundError(f"No event files found in {self.events_path}")
            self.event_file = event_files[-1]
        else:
            self.event_file = self.events_path

        self.ea = event_accumulator.EventAccumulator(str(self.event_file), size_guidance={"scalars": 0})
        self.ea.Reload()
    
    def get_data(self, tag: str) -> pd.DataFrame:
        """
        Returns a DataFrame with columns: step, tag
        """
        events = self.ea.Scalars(tag)
        df = pd.DataFrame(
            {
            "step":[e.step for e in events],
            "value":[e.value for e in events],
            }
        )
        col_name = tag.lower().replace("/", "_").replace(" ", "_")
        df = df.rename(columns={"value": col_name})

        return df

    def append_metrics(self, entropy_tag: str = "Policy/Entropy", policy_loss_tag: str = "Losses/Policy Loss", value_loss_tag: str = "Losses/Value Loss") -> pd.DataFrame:
        """
        Read entropy, policy and value loss from events and merge it to the training DataFrame on step
        
        """

        entropy_df = self.get_data(entropy_tag)
        policy_loss_df = self.get_data(policy_loss_tag)
        value_loss_df = self.get_data(value_loss_tag)

        metrics_df = entropy_df.merge(policy_loss_df, on="step", how="outer")
        metrics_df = metrics_df.merge(value_loss_df, on="step", how="outer")

        final_df = self.train_df.merge(metrics_df, on="step", how="left")
        self.train_df = final_df

        self.train_df.to_csv(self.train_csv_path, index=False)

