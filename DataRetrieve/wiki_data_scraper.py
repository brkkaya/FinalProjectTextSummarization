from io import StringIO
from re import search
from DataRetrieve.wiki_data_parser import DataParser
from src.services.base_service import BaseService
from mediawiki import MediaWiki
import csv, codecs
from src.services.path_provider import GLobalPathProvider


class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = StringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# TODO Create a module for later
class DataScraper(BaseService):
    def __init__(self, parser=DataParser) -> None:
        self.parser = parser
        self.log.info("Test Scraper Init")
        self.wikipedia = MediaWiki(lang="tr",)

        self.csv_file = open("wiki_data.csv", "a", encoding="utf-8")
        self.csv = csv.writer(self.csv_file, delimiter=",")
        if self.config.write_header:
            self.csv.writerow(["Summary", "Page Content"])
        self.dictionary = open("wiki_dictionary.txt", "a", encoding="utf-8")
        self.dict = {}

    def _run(self):
        
        self.log.info("Scraper Run")
        self.search_page("Roma İmparatorluğu",0)
        

    def read_page(self, page_title: str):
        page = self.wikipedia.page(page_title)
        summary = page.summary
        if len(summary) >= 300:
            self.csv.writerow([summary, page.content])
            self.csv_file.flush()
            self.log.info(f"Page exported: {page_title}")
        return page

    def search_page(self, init_page: str, depth: int = 0):
        # TODO Make this method recursive, this method will add every sublinks to a list in the and, it will pop itself.
        if depth == 20:
            return
        try:
            page = self.read_page(init_page)
        except:
            return
        for x in page.links:
            if x not in self.dict:
                self.dict[x] = 1
                self.dictionary.write(x + "\n")
                self.dictionary.flush()
                self.search_page(x, depth + 1)
