from pathlib import Path
from cityscholar.config import AppConfig
from cityscholar.agent import CityScholarAgent
from cityscholar.multiagent import MultiAgentHub
import logging
import time

logger = logging.getLogger("cityscholar.demo")


def main():
    config = AppConfig.from_env(storage_dir=Path('storage'))
    agent = CityScholarAgent(config)

    hub = MultiAgentHub()

    # Reader agent: performs retrieval QA using the CityScholarAgent
    def reader(task: dict):
        q = task.get('query') or task.get('text') or ''
        resp = agent.answer(q, top_k=4)
        return {'content': resp.content, 'markdown': resp.to_markdown()}

    # Summarizer agent: try to use LLMClient for high-quality summary, fallback to local summarizer
    def summarizer(task: dict):
        texts = task.get('texts', [])
        if not texts:
            return {'summary': 'no texts provided'}
        # build a stronger local fallback using agent's TF-IDF summarizer
        try:
            fallback_text = agent._local_summarize_texts(texts, n_sentences=5)
        except Exception:
            fallback_text = '\n'.join(texts)[:800]

        joined = '\n\n'.join(t for t in texts)
        prompt = (
            "请基于以下检索证据生成一段中文摘要（3-6 句），要包含研究主题、关键结论和可引用的证据线索：\n\n" + joined
        )

        # use agent.llm if available; provide the stronger fallback
        try:
            if getattr(agent, 'llm', None):
                content = agent.llm.generate(prompt, fallback_text)
                return {'summary': content}
        except Exception:
            pass

        return {'summary': fallback_text}

    hub.register('reader', reader)
    hub.register('summarizer', summarizer)

    query = 'user-level differential privacy contextual bandits'
    logger.info('Broadcasting query to readers...')
    # broadcast to all; reader will respond, summarizer will also get the task but ignore
    results = hub.broadcast({'query': query})

    # collect reader outputs
    reader_texts = []
    for name, res in results.items():
        if isinstance(res, dict) and 'content' in res:
            reader_texts.append(res['content'])
            logger.info(f'[{name}] returned content length {len(res["content"])}')

    # coordinate summarizer
    summary_results = hub.coordinate({'texts': reader_texts}, participants=['summarizer'])
    logger.info('\nSummary results:')
    logger.info(summary_results.get('summarizer'))


if __name__ == '__main__':
    main()
