import json
import os
from utils.config import CONFIG


def get_base_data_sources():
    """Returns the initial categorized data sources as a dictionary."""
    return {
        "scrapable_or_direct_download": [
            {
                "name": "data.gov.in",
                "domain": "All domains - Agriculture, Energy, Climate, Rural",
                "website": "https://data.gov.in",
                "data_tip": "Open datasets from Indian ministries and government programs"
            },
            {
                "name": "Our World in Data",
                "domain": "Energy, Agriculture, Climate, Global Development",
                "website": "https://ourworldindata.org",
                "data_tip": "Downloadable CSVs, visualizations, curated datasets"
            },
            {
                "name": "IRENA (International Renewable Energy Agency)",
                "domain": "Renewable Energy Policies & Statistics",
                "website": "https://www.irena.org",
                "data_tip": "Global and regional renewable energy data & reports"
            },
            {
                "name": "ICAR (Indian Council of Agricultural Research)",
                "domain": "Agriculture, Sustainable Farming",
                "website": "https://icar.org.in",
                "data_tip": "Research papers, annual reports, project documents, and agri-statistics"
            },
            {
                "name": "MNRE (Ministry of New and Renewable Energy)",
                "domain": "Renewable & Sustainable Energy",
                "website": "https://mnre.gov.in",
                "data_tip": "Annual reports, policy documents, state-level dashboards"
            },
            {
                "name": "NITI Aayog",
                "domain": "Sustainable Development Goals (SDGs), Agriculture",
                "website": "https://niti.gov.in",
                "data_tip": "SDG India Index, agricultural planning, climate resilience docs"
            },
            {
                "name": "UNEP (United Nations Environment Programme)",
                "domain": "Sustainability, Energy, Environment",
                "website": "https://www.unep.org",
                "data_tip": "Reports, climate energy dashboards"
            },
            {
                "name": "WRI (World Resources Institute)",
                "domain": "Climate, Land, Energy, Water, Sustainability",
                "website": "https://www.wri.org",
                "data_tip": "Global energy dashboards, sustainable development data"
            }
        ],
        "api_access_required": [
            {
                "name": "CORE",
                "domain": "Open Access Research Papers",
                "website": "https://core.ac.uk",
                "data_tip": "Great for building academic knowledge bases for RAG"
            },
            {
                "name": "arXiv",
                "domain": "Open Access Preprints – AI, Agriculture, Environment",
                "website": "https://arxiv.org",
                "data_tip": "Use arXiv API to fetch paper metadata and abstracts (supports query by subject and keyword)"
            }
        ]
    }


def get_additional_sources():
    """Returns additional specific links for ICAR, MNRE, and DAFW."""
    return {
        "ICAR": [
            "https://icar.org.in/annual-report",
            "https://icar.org.in/publication-catalogue",
            "https://epubs.icar.org.in/",
            "https://icar.org.in/abstracting-journals",
            "https://icar.org.in/ITK",
            "https://icar.org.in/e-books",
            "https://icar.org.in/official-language-magazines",
            "https://icar.org.in/other-reports",
            "https://icar.org.in/indian-horticulture",
            "https://icar.org.in/indian-farming",
            "https://icar.org.in/swachhta-hi-sewa",
        ],
        "MNRE": [
            "https://mnre.gov.in/en/akshay-urja-magazine/",
            "https://mnre.gov.in/en/annual-report/",
            "https://mnre.gov.in/en/renewable-energy-statistics/",
            "https://mnre.gov.in/en/other-report/",
        ],
        "DAFW": [
            "https://agriwelfare.gov.in/en/Agricultural_Statistics_at_a_Glance",
            "https://desagri.gov.in/statistics-type/latest-minimum-support-price-msp-statement/",
            "https://desagri.gov.in/statistics-type/advance-estimates/",
            "https://desagri.gov.in/statistics-type/five-year-estimates/",
            "https://desagri.gov.in/statistics-type/normal-estimates/",
            "https://desagri.gov.in/statistics-type/minor-crop-estimates/",
            "https://agriwelfare.gov.in/en/AgricultureEstimates",
            "https://agriwelfare.gov.in/en/StatHortEst",
            "https://agriwelfare.gov.in/en/PublicationReports",
            "https://agriwelfare.gov.in/en/Horticulturereports",
            "https://agriwelfare.gov.in/en/G20OUTCOME",
            "https://agriwelfare.gov.in/en/Annual",
            "https://agriwelfare.gov.in/en/Other",
            "https://agriwelfare.gov.in/en/Presentation",
            "https://agriwelfare.gov.in/en/DocAgriContPlan",
            "https://agriwelfare.gov.in/en/Reservoir",
            "https://agriwelfare.gov.in/en/Rain",
            "https://agriwelfare.gov.in/en/CropSituation",
            "https://agriwelfare.gov.in/en/weather-watch",
            "https://agriwelfare.gov.in/en/docarchive",
            "https://agriwelfare.gov.in/en/Handbook_of_Work_Allocation",
            "https://agriwelfare.gov.in/en/FarmWelfare",
            "https://agriwelfare.gov.in/en/Major",
            "https://agriwelfare.gov.in/en/Guide",
            "https://upag.gov.in/primary-estimate-report",
            "https://upag.gov.in/primary-estimate-report?tab=Market+Intelligence",
            "https://upag.gov.in/primary-estimate-report?tab=Prices",
            "https://upag.gov.in/primary-estimate-report?tab=Procurement+and+Stock",
            "https://upag.gov.in/primary-estimate-report?tab=Trade",
            "https://upag.gov.in/primary-estimate-report?tab=Miscellaneous",
            "https://desagri.gov.in/aer-report/",
            "https://agriwelfare.gov.in/en/Acts",
            "https://desagri.gov.in/document-report-category/agriculture-statistics-at-a-glance/",
            "https://desagri.gov.in/document-report-category/pocket-book-of-agricultural-statistics/",
            "https://desagri.gov.in/document-report-category/agricultural-situation-in-india/",
            "https://desagri.gov.in/document-report-category/agricultural-prices-in-india/",
            "https://desagri.gov.in/document-report-category/agriculture-wages-in-india/",
            "https://desagri.gov.in/document-report-category/state-of-indian-agriculture/",
            "https://desagri.gov.in/document-report-category/crop-complex-for-the-year/",
            "https://desagri.gov.in/document-report-category/selected-zone-tehsil-district-block-year-wise/",
            "https://desagri.gov.in/document-report-category/farm-harvest-prices-of-principle-crops-in-india/",
            "https://desagri.gov.in/document-report-category/land-use-statistics-at-a-glance/",
            "https://desagri.gov.in/document-report-category/monthly-bulletin/"
        ]
    }


def append_additional_sources(base_sources, additional_sources):
    """
    Append additional link sets into the 'scrapable_or_direct_download' list
    with metadata.
    """
    for org, links in additional_sources.items():
        base_sources["scrapable_or_direct_download"].append({
            "name": org,
            "domain": "Agriculture / Renewable Energy / Statistics",
            "website": links[0],  # first link for reference
            "data_tip": f"Multiple downloadable datasets and reports under {org}",
            "sub_links": links
        })


def save_json(data, filepath):
    """Save dictionary as pretty JSON to given filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Create folder(s) if missing
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load JSON file and return as dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # Get base data source dictionaries
    data_sources = get_base_data_sources()
    
    # Get additional specific sources
    additional = get_additional_sources()
    
    # Append additional sources into base dictionary
    append_additional_sources(data_sources, additional)
    
    # Save to a JSON file
    json_path = CONFIG["data_collection_paths"]["data_sources_json"]
    save_json(data_sources, json_path)
    
    print(f"Data sources saved successfully to {json_path}")


if __name__ == "__main__":
    main()