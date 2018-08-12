from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from library.dataset_readers.imdb_review_reader import IMDBLanguageModelingReader


class TestIMDBReader(AllenNlpTestCase):
    # Hard-coded excerpts from the corpus for smoke testing.
    INSTANCE_0 = {
            "id": 0,
            "source": [START_SYMBOL, "Resumption", "of", "the", "session", END_SYMBOL],
            "target": [START_SYMBOL, "Reprise", "de", "la", "session", END_SYMBOL]
    }
    INSTANCE_1 = {
            "id": 1,
            "source": [START_SYMBOL, "I", "declare", "resumed", "the", "session", "of", "the",
                       "European", "Parliament", "adjourned", "on", "Friday", "17", "December",
                       "1999", ",", "and", "I", "would", "like", "once", "again", "to", "wish",
                       "you", "a", "happy", "new", "year", "in", "the", "hope", "that",
                       "you", "enjoyed", "a", "pleasant", "festive", "period", ".", END_SYMBOL],
            "target": [START_SYMBOL, "Je", "déclare", "reprise", "la", "session", "du",
                       "Parlement", "européen", "qui", "avait", "été", "interrompue", "le",
                       "vendredi", "17", "décembre", "dernier", "et", "je", "vous", "renouvelle",
                       "tous", "mes", "vux", "en", "espérant", "que", "vous", "avez", "passé",
                       "de", "bonnes", "vacances", ".", END_SYMBOL]
    }

    INSTANCE_7 = {
            "id": 7,
            "source": [START_SYMBOL, "Madam", "President", ",", "on", "a", "point", "of", "order",
                       ".", END_SYMBOL],
            "target": [START_SYMBOL, "Madame", "la", "Présidente", ",", "c'", "est", "une",
                       "motion", "de", "procédure", ".", END_SYMBOL]
    }

    DATASET_PATH = 'tests/fixtures/smoke.jsonl'

    def test_read_from_file(self):
        # pylint: disable=R0201
        reader = IMDBLanguageModelingReader()
        dataset = reader.read(TestIMDBReader.DATASET_PATH)
        instances = ensure_list(dataset)

        assert len(instances) == 10
        fields = instances[0].fields
        assert [t.text for t in fields["source"].tokens] == TestIMDBReader.INSTANCE_0["source"]
        assert [t.text for t in fields["target"].tokens] == TestIMDBReader.INSTANCE_0["target"]
        fields = instances[1].fields
        assert [t.text for t in fields["source"].tokens] == TestIMDBReader.INSTANCE_1["source"]
        assert [t.text for t in fields["target"].tokens] == TestIMDBReader.INSTANCE_1["target"]
        fields = instances[7].fields
        assert [t.text for t in fields["source"].tokens] == TestIMDBReader.INSTANCE_7["source"]
        assert [t.text for t in fields["target"].tokens] == TestIMDBReader.INSTANCE_7["target"]
