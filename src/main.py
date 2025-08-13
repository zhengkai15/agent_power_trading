"""
Main entry point for the application.
"""
from src.utils.logging import log
from src.training.train_predictor import train_price_predictor
from src.training.train_agent import train_rl_agent
import argparse

def main():
    """
    Main function to run the application.
    """
    parser = argparse.ArgumentParser(description="Electricity Trading Agent")
    parser.add_argument(
        "action",
        choices=["train_predictor", "train_agent", "run_trader"],
        help="Action to perform",
    )
    args = parser.parse_args()

    log.info(f"Starting action: {args.action}")

    if args.action == "train_predictor":
        train_price_predictor()
    elif args.action == "train_agent":
        train_rl_agent()
    elif args.action == "run_trader":
        # Placeholder for running the trader in inference mode
        log.info("Running the trader is not yet implemented in main.")
        pass
    else:
        log.error(f"Unknown action: {args.action}")

    log.info(f"Action {args.action} finished.")

if __name__ == "__main__":
    main()
