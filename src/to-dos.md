


To Dos
------
1 - Add function for Hybrid-Search (Hybrid search with weights is working)
2 - check if ingesting multiple files works
3 - Instantiate 'metadata filtering' 
  - works but needs to become a dict rather than a list (of keywords at the moment)
  - should have this structure:
    {
        "keywords: ["keyword1",.... "keyword5"],
        "title: "slepping children for the night",
        "publishing_year": 2015,
        ....
        ....
        "page_count": 6
        
    }
4 - Add a memory component - so that it can be a conversation rather than a point and shot.
- Introduce Multi-modal chunking
- Evaluation of system (precision and recall)

Nice-to-have
===========
1) - Streamlit?
2) - Prompt manager in Jinja2

