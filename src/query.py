"""CLI: python src/query.py 'What is earned value?' --source-doc pmbok"""
import argparse
import logging

from service import get_service, source_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question', nargs='?')
    parser.add_argument('--source-doc', help='The --doc-name used at ingestion.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        service = get_service()
    except Exception as exc:
        parser.exit(1, f'Initialization failed: {exc}\n')

    def ask(question):
        try:
            result = service.answer(question, source_doc=args.source_doc)
            print('\n' + result.answer)
            for source in result.sources:
                print(
                    f'[{source["source_id"]}] {source_label(source["metadata"])} ({source["metadata"]["type"]})')
        except Exception as exc:
            print(f'Could not answer: {exc}')
            return False
        return True

    if args.question:
        if not ask(args.question):
            parser.exit(1)
        return
    print("Ask a question, or type 'quit'. Use a new process after re-ingestion.")
    while True:
        try:
            question = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {'quit', 'exit'}:
            break
        if question:
            ask(question)


if __name__ == '__main__':
    main()
