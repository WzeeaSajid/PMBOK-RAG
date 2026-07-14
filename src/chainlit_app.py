"""Run from the repository root: chainlit run src/chainlit_app.py"""
import logging

import chainlit as cl

from service import get_service

log = logging.getLogger(__name__)


async def warm_app():
    log.info('Warming models and search index before accepting chats...')
    await cl.make_async(get_service)()
    log.info('RAG startup complete. Resources will be reused across chats.')


# Older Chainlit releases can fall back to first-chat initialization.
if hasattr(cl, 'on_app_startup'):
    cl.on_app_startup(warm_app)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set('service', None)
    status = cl.Message(content='Loading the document index and local models…')
    await status.send()
    try:
        # Model/HTTP/index initialization all run outside the event loop.
        service = await cl.make_async(get_service)()
        cl.user_session.set('service', service)
        await cl.Message(content='Ready. Ask about your indexed documents. Sources include document and page details.').send()
    except Exception as exc:
        log.exception('Initialization failed')
        await cl.Message(content=f'Could not initialize: {exc}').send()
    finally:
        await status.remove()


@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()
    if not question:
        return
    service = cl.user_session.get('service')
    if service is None:
        await cl.Message(content='The pipeline is not ready. Resolve the startup error and refresh.').send()
        return
    try:
        result = await cl.make_async(service.answer)(question)
        labels = '\n'.join(f'[{s["source_id"]}] {s["label"]}' for s in result.sources)
        await cl.Message(content=result.answer + ('\n\n' + labels if labels else '')).send()
        for source in result.sources:
            async with cl.Step(name=f'[{source["source_id"]}] {source["label"]}', type='retrieval') as step:
                step.output = source.get('full_text', source['text'])
    except Exception as exc:
        log.exception('Question failed')
        await cl.Message(content=f'Could not answer: {exc}').send()
