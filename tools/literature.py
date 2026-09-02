import httpx

from models.hypothesis import LiteratureResult, Paper


OPENALEX_URL = "https://api.openalex.org/works"


def reconstruct_abstract(
    inverted_index: dict[str, list[int]] | None,
) -> str:
    """
    Convert OpenAlex's inverted abstract index into normal text.
    """

    if not inverted_index:
        return ""

    words: list[tuple[int, str]] = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort(key=lambda item: item[0])

    return " ".join(word for _, word in words)


def search_literature(
    query: str,
    limit: int = 5,
) -> LiteratureResult:
    """
    Search OpenAlex for papers related to a research question.
    """

    params = {
        "search": query,
        "per-page": limit,
    }

    response = httpx.get(
        OPENALEX_URL,
        params=params,
        timeout=30.0,
    )

    response.raise_for_status()

    data = response.json()

    papers = []

    for work in data.get("results", []):
        paper = Paper(
            id=work.get("id", ""),
            title=work.get("display_name", "Untitled"),
            abstract=reconstruct_abstract(
                work.get("abstract_inverted_index")
            ),
            publication_year=work.get("publication_year"),
            doi=work.get("doi"),
            cited_by_count=work.get("cited_by_count", 0),
        )

        papers.append(paper)

    return LiteratureResult(papers=papers)