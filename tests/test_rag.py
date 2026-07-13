"""Retrieval eval for rag/contract_search.py — a labeled query -> expected clause
set, checked for top-1 hit rate. This is what makes it a measured RAG component
rather than a demo that's never actually been checked for accuracy: every query
below is phrased the way an SLA-lookup step would ask it (not copy-pasted from
the contract text itself), so this tests real semantic matching, not string overlap.
"""
import pytest

from rag.contract_search import search

# (query, supplier_id, expected_clause_id)
LABELED_QUERIES = [
    ("what is the penalty for a late delivery shipment", "acme-logistics", "3.2"),
    ("how quickly must the supplier report an expected delay", "acme-logistics", "3.1"),
    ("customer rejected goods due to quality defects", "acme-logistics", "4.2"),
    ("supplier cannot meet order volume, capacity shortfall", "acme-logistics", "5.1"),
    ("shipment held at customs due to missing paperwork", "acme-logistics", "6.1"),
    ("invoice does not match the purchase order price", "acme-logistics", "7.1"),
    ("what is the penalty for a late delivery shipment", "northwind-parts", "4.1"),
    ("customer rejected goods due to quality defects", "northwind-parts", "5.1"),
    ("shipment held at customs due to missing paperwork", "northwind-parts", "7.1"),
    ("invoice does not match the purchase order price", "northwind-parts", "8.1"),
]


@pytest.mark.parametrize("query,supplier_id,expected_clause_id", LABELED_QUERIES)
def test_top_hit_matches_expected_clause(query, supplier_id, expected_clause_id):
    results = search(query, supplier_id=supplier_id, top_k=1)
    assert results, f"no results for {query!r} / {supplier_id}"
    assert results[0]["clause_id"] == expected_clause_id, (
        f"query {query!r} on {supplier_id}: expected clause {expected_clause_id}, "
        f"got {results[0]['clause_id']} ({results[0]['title']!r}, score={results[0]['score']:.3f})"
    )


def test_supplier_filter_excludes_other_suppliers():
    results = search("late delivery penalty", supplier_id="acme-logistics", top_k=5)
    assert all(r["supplier_id"] == "acme-logistics" for r in results)
