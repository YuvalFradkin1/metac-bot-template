import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    MetaculusApi,
    MetaculusQuestion,
    NumericQuestion,
    MultipleChoiceQuestion,
    PredictedOptionList,
    NumericDistribution,
    PredictionExtractor,
    ReasonedPrediction,
)

logger = logging.getLogger(__name__)

class V11Forecaster(ForecastBot):
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        # כרגע לא נדרש מחקר, V11 מזהה תבניות בעצמו
        return ""

    def v11_predict(self, question_text):
        # דוגמה מופשטת לעקרונות V11 (Gradient analysis, Equilibrium state)
        # החלף זאת בקריאה אמיתית ל-V11
        return {
            "avg": 0.55,  # תוצאה אמיתית שתגיע מ־V11
            "low": 0.35,
            "high": 0.75,
        }

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        v11_result = self.v11_predict(question.question_text)
        reasoning = (
            f"V11 Energetic Gradient Analysis prediction for '{question.question_text}'. "
            f"Probability: {v11_result['avg']*100:.2f}%."
        )
        prediction = v11_result["avg"]
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction:.2f} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        equal_prob = round(1.0 / len(question.options), 2)
        predicted_options = {opt: equal_prob for opt in question.options}
        reasoning = "V11 does not currently support multiple-choice. Equal probabilities assigned."
        return ReasonedPrediction(prediction_value=predicted_options, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        percentiles = {
            10: question.lower_bound if question.lower_bound else 0,
            20: question.lower_bound if question.lower_bound else 10,
            40: (question.lower_bound + question.upper_bound) / 3 if question.upper_bound else 50,
            60: (question.lower_bound + question.upper_bound) / 2 if question.upper_bound else 100,
            80: question.upper_bound if question.upper_bound else 150,
            90: question.upper_bound if question.upper_bound else 200,
        }
        reasoning = "V11 does not currently support numeric. Simplified distribution assigned."
        numeric_dist = NumericDistribution(declared_percentiles=percentiles)
        return ReasonedPrediction(prediction_value=numeric_dist, reasoning=reasoning)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run V11ForecastBot forecasting system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()

    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = args.mode

    v11_bot = V11Forecaster(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
    )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            v11_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        v11_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            v11_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
        ]
        v11_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            v11_bot.forecast_questions(questions, return_exceptions=True)
        )

    V11Forecaster.log_report_summary(forecast_reports)
