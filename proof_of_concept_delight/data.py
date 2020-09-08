from __future__ import annotations

from typing import NamedTuple, List
import datetime
import csv

def _parse(date: str) -> datetime.date:
    y, m, d = date.strip().split(" ")[0].split("/")
    return datetime.date(int(y), int(m), int(d))

class NewsArticle(NamedTuple):
    article_id: int
    publish_date: datetime.date
    article_source_link: str
    title: str
    subtitle: str
    text: str

    @staticmethod
    def from_row(row: List[str]) -> NewsArticle:
        return NewsArticle(
            article_id=int(row[0]),
            publish_date=_parse(row[1]),
            article_source_link=row[2],
            title=row[3],
            subtitle=row[4],
            text=row[5]
        )

    @property
    def site(self) -> str:
        url = self.article_source_link
        address = url.split("//")[1]
        domain = address.split("/")[0]
        return domain


def load_articles(fn: str) -> List[NewsArticle]:
    with open(fn, encoding='unicode_escape') as f:
        reader = csv.reader(f)
        # skip header row
        next(reader)
        return [NewsArticle.from_row(row) for row in reader]