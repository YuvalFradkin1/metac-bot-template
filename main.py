import argparse
import asyncio
import logging
import os
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

# הגדרת OpenAI API key מתוך משתני סביבה
openai.api_key = os.getenv("OPENAI_API_KEY")

class V11Forecaster(ForecastBot):
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        return ""  # אין צורך במחקר נוסף כרגע, ההמרה מתבצעת באמצעות OpenAI

    # המרת טקסט לסדרה מספרית באמצעות OpenAI
    async def text_to_embedding(self, text):
        response = await openai.Embedding.acreate(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response["data"][0]["embedding"]
        return embedding

    # שימוש אמיתי ב־V11 על הנתונים המספריים
    async def v11_predict(self, numeric_series):
        # כאן תבצע קריאה אמיתית ל־V11 שלך
        # לצורך ההדגמה בלבד, אני מחשב הסתברות פשוטה
        energy_level = sum(numeric_series) / len(numeric_series)
        probability = max(0, min(1, (energy_level % 1)))  # הדגמה של שימוש אנרגטי
        return {
            "avg": probability,
            "low": max(0, probability - 0.1),
            "high": min(1, probability + 0.1),
        }

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        embedding = await self.text_to_embedding(question.question_text)
        v11_result = await self.v11_predict(embedding)

        reasoning = (
            f"V11 Energetic prediction using OpenAI embeddings: "
            f"Probability: {v11_result['avg'] * 100:.2f}%."
        )

        prediction = v11_result["avg"]
        logger.info(f"Forecasted {question.page_url} as {prediction:.2f}\n{reasoning}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        equal_prob = round(1.0 / len(question.options), 2)
        predicted_options = {opt: equal_prob for opt in question.options}
        reasoning = "Multiple-choice not yet fully supported by V11."
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
        numeric_dist = NumericDistribution(declared_percentiles=percentiles)
        reasoning = "Numeric questions not yet fully supported by V11."
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
