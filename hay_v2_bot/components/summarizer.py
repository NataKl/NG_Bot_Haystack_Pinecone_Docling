from openai import OpenAI


class FileSummarizer:
    def __init__(self, openai_client: OpenAI, model: str) -> None:
        self.openai_client = openai_client
        self.model = model

    def summarize_one_sentence(self, text: str) -> str:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты помощник по анализу документов. "
                        "Возвращай строго одно короткое предложение-резюме на русском языке."
                    ),
                },
                {"role": "user", "content": text[:7000]},
            ],
        )
        summary = (completion.choices[0].message.content or "").strip()
        if not summary:
            return "Документ обработан и сохранен, в нем содержится структурированная текстовая информация."
        return summary.split("\n")[0].strip()
