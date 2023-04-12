"""Logseq reader.

Load the pages, journals, assets, and tags from a Logseq vault.
"""

import os
from pathlib import Path
from typing import Any, List

from langchain.docstore.document import Document as LCDocument

from llama_index.readers.base import BaseReader
from llama_index.readers.file.markdown_parser import MarkdownParser
from llama_index.readers.schema.base import Document


class LogseqReader(BaseReader):
    """Utilities for loading data from an Logseq Vault.

    Args:
        input_dir (str): Path to the vault.

    """

    def __init__(self, input_dir: str):
        """Init params."""
        self.input_dir = Path(input_dir)
        self.pages_dir = self.input_dir / "pages"

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the pages directory."""
        docs: List[str] = []
        for dirpath, _, filenames in os.walk(self.pages_dir):
            for filename in filenames:
                if filename.endswith(".md"):
                    filepath = os.path.join(dirpath, filename)
                    content = MarkdownParser().parse_file(Path(filepath))
                    docs.extend(content)
        return [Document(d) for d in docs]

    def load_langchain_documents(self, **load_kwargs: Any) -> List[LCDocument]:
        """Load data in LangChain document format."""
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]
