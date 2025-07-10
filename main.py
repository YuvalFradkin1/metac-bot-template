import argparse
import asyncio
import logging
import os
import json
import time
from typing import Literal
import openai
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
    NumericDistribution,
    ReasonedPrediction,
)

logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

PREDICTIONS_FILE = "past_predictions.json"
LAST_UPDATE_FILE = "last_accuracy_update.txt"
UPDATE_INTERVAL = 48 * 60 * 60  # 48 שעות בשניות

class V11Forecaster(ForecastBot):
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.past_predictions = self.load_past_predictions()

    def load_past_predictions(self):
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, "r") as f:
                return json.load(f)
        return {}

    def save_past_predictions(self):
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(self.past_predictions, f)

    async def run_research(self, question: MetaculusQuestion) -> str:
        return ""

    async def text_to_embedding(self, text):
        response = await openai.Embedding.acreate(
            input=text, model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]

    async def v11_predict(self, numeric_series):
        energy_level = sum(numeric_series) / len(numeric_series)
        adjustment = self.get_adjustment_factor()
        probability = max(0, min(1, (energy_level % 1) * adjustment))
        return {
            "avg": probability,
            "low": max(0, probability - 0.1),
            "high": min(1, probability + 0.1),
        }

    def get_adjustment_factor(self):
        if not self.past_predictions:
            return 1.0
        accuracy = sum(self.past_predictions.values()) / len(self.past_predictions)
        return 1.0 + (accuracy - 0.5)

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        embedding = await self.text_to_embedding(question.question_text)
        v11_result = await self.v11_predict(embedding)
        reasoning = (
            f"V11 prediction adjusted by past accuracy. Probability: {v11_result['avg']*100:.2f}%."
        )
        prediction = v11_result["avg"]
        logger.info(f"Forecasted {question.page_url} as {prediction:.2f}\n{reasoning}")

        self.past_predictions[str(question.id)] = prediction
        self.save_past_predictions()

        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prob_per_option = round(1.0 / len(question.options), 2)
        predicted_options = {opt: prob_per_option for opt in question.options}
        reasoning = "Multiple-choice not yet fully supported by V11, assigning equal probability."
        logger.info(f"Forecasted multiple-choice question {question.page_url} equally.")
        return ReasonedPrediction(prediction_value=predicted_options, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        lower = question.lower_bound or 0
        upper = question.upper_bound or lower + 100
        midpoint = (lower + upper) / 2
        percentiles = {
            10: lower,
            20: lower + (midpoint - lower) * 0.5,
            40: midpoint,
            60: midpoint,
            80: midpoint + (upper - midpoint) * 0.5,
            90: upper,
        }
        numeric_dist = NumericDistribution(declared_percentiles=percentiles)
        reasoning = "Numeric questions not yet fully supported by V11, using default numeric distribution."
        return ReasonedPrediction(prediction_value=numeric_dist, reasoning=reasoning)

    async def update_past_accuracy(self):
        if not os.path.exists(LAST_UPDATE_FILE):
            with open(LAST_UPDATE_FILE, "w") as f:
                f.write(str(time.time()))
            logger.info("Initial run detected, skipping accuracy update this time.")
            return

        with open(LAST_UPDATE_FILE, "r") as f:
            last_update = float(f.read())

        current_time = time.time()
        if current_time - last_update < UPDATE_INTERVAL:
            logger.info("Accuracy update skipped (updated less than 48 hours ago).")
            return

        all_questions = await MetaculusApi().get_benchmark_questions(num_of_questions_to_return=100)
        resolved_questions = [q for q in all_questions if q.is_resolved]

        for q in resolved_questions:
            q_id_str = str(q.id)
            if q_id_str in self.past_predictions:
                try:
                    correct_answer = float(q.resolution)
                    predicted_answer = self.past_predictions[q_id_str]
                    accuracy = 1 - abs(correct_answer - predicted_answer)
                    self.past_predictions[q_id_str] = accuracy
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse resolution for question ID {q_id_str}")

        self.save_past_predictions()
        with open(LAST_UPDATE_FILE, "w") as f:
            f.write(str(current_time))
        logger.info("Past predictions accuracy updated successfully.")

async def run_multiple_tournaments(bot, tournament_ids):
    for index, tid in enumerate(tournament_ids):
        if index > 0:
            logger.info("Waiting 1 minute before starting next tournament to avoid blocking.")
            await asyncio.sleep(60)  # השהייה של דקה אחת בין טורנירים
        await bot.forecast_on_tournament(tid, return_exceptions=True)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run V11ForecastBot forecasting system")
    parser.add_argument("--mode", type=str, choices=["tournament"], default="tournament")
    args = parser.parse_args()

    v11_bot = V11Forecaster(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
    )

    logger.info("Starting automatic accuracy update check...")
    asyncio.run(v11_bot.update_past_accuracy())

    TOURNAMENT_IDS = [
        32721, 32726, 32775, 32725, 32722, 32564, 3411, 1756, 1998, 1886
    ]

    logger.info("Starting forecasts on multiple tournaments...")
    asyncio.run(run_multiple_tournaments(v11_bot, TOURNAMENT_IDS))
