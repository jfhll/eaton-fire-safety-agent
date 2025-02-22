import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

urls = [
    "https://www.nature.com/articles/s41370-020-00267-4",
    "https://www.nature.com/articles/s41467-023-43101-9",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11562715/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9985454/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10094253/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11268558/",
    "https://www.epa.gov/children/protecting-childrens-health-during-and-after-natural-disasters-wildfires-volcanic-ash",
    "https://www.epa.gov/sites/default/files/2018-11/documents/protecting-children-from-wildfire-smoke-and-ash.pdf",
    "https://www.epa.gov/wildfire-smoke-course/fire-or-smoke-event-plan",
    "https://www.epa.gov/emergencies-iaq/wildfires-and-indoor-air-quality-iaq",
    "https://www.epa.gov/sites/default/files/2019-07/documents/wagner_cdph-oaq_health_impacts_of_particles_and_gases_emitted_by_wildfires_tagged.pdf",
    "https://www.epa.gov/asbestos/how-many-samples-miscellaneous-material-or-nonfriable-suspected-material-must-be-taken",
    "https://www.epa.gov/expobox/exposure-assessment-tools-chemical-classes-other-organics",
    "https://www.epa.gov/pfas/interim-guidance-destruction-and-disposal-pfas-and-materials-containingpfas",
    "https://www.ecos.org/wp-content/uploads/2024/06/2024-ECOS-PFAS-Standards-Paper-Update.pdf",
    "https://www.remediation-technology.com/articles/368-epa-touts-record-wildfire-cleanup-progress-as-communities-push-back-on-toxic-waste-sites",
    "https://jselabs.com/asbestos-analysis/how-to-take-an-asbestos-sample/",
    "https://www.partneresi.com/resources/articles/what-is-an-asbestos-survey/",
    "https://bilbau.ch/en/article/covered-inter-laboratory-test-high-error-rate-asbestos-laboratories",
    "https://spokanecleanair.org/fire-damaged-structures-asbestos-a-quick-review-of-initial-steps/",
    "https://lexscientific.com/Asbestos-Sampling-Guide.htm",
    "https://www.oracleasbestos.com/blog/surveys/test-for-asbestos/",
    "https://lcslaboratory.com/never-mix-materials-for-asbestos-testing/",
    "https://archive.epa.gov/region9/toxic/web/pdf/epa-ert-asbestos-sampling-sop-2015.pdf",
    "https://www.mpaasbestosremoval.com.au/is-asbestos-testing-reliable/",
    "https://www.science.org/doi/10.1126/sciadv.adh8263",
    "https://nwfirescience.org/sites/default/files/publications/sciadv.adh8263.pdf",
    "https://pubs.acs.org/doi/10.1021/acsestair.4c00259",
    "https://acp.copernicus.org/articles/23/12441/2023/",
    "https://fsri.org/research/heat-transfer-and-fire-damage-patterns-walls-fire-model-validation",
    "https://cdn.prod.website-files.com/64b9346df4252df1681cba3e/64c5f47fde0296f89f32fa25_cyanide_poisonings_of_providence_firefighters.pdf",
    "https://www.cdc.gov/wildfires/safety/how-to-safely-stay-safe-during-a-wildfire.html",
    "https://www.npr.org/sections/health-shots/2024/07/26/5049828/wildfire-smoke-health-risks-safety-air-quality-index",
    "https://www.nytimes.com/wirecutter/reviews/best-respirator-mask/",
    "https://clark.wa.gov/sites/default/files/dept/files/public-health/wildfire-smoke/DOH_Wildfire_Smoke_Face_Masks_Factsheet.pdf",
    "https://www.airnow.gov/sites/default/files/2022-01/childrens-health-wildfire-smoke-workshop-recommendations.pdf",
    "https://warddiesel.com/wp-content/uploads/2023/09/Hierarchy-of-contamination-control-in-the-fire-service-Review-of-exposure-control-options-to-reduce-cancer-risk.pdf",
    "https://www.oregon.gov/osfm/healthandsafety/Documents/Hierarchy-of-contamination-control-in-the-fire-service-Review-of-exposure-control-options-to-reduce-cancer-risk.pdf",
    "https://www.latimes.com/environment/story/2025-01-30/la-wildfires-hazardous-waste-cleanup",
    "https://laist.com/news/climate-environment/study-tracks-health-impacts-pollution-la-wildfires",
    "https://laist.com/brief/news/climate-environment/researchers-tested-sandboxes-street-dust-lead-eaton-fire",
    "https://www.kcrw.com/culture/shows/good-food/fire-soil-safety-lunar-new-year-china-dishes/eaton-palisades-fire-soil-ash-residue-fallout-danger-garden-fruit-vegetables",
    "https://abc7.com/post/toxic-dangers-linger-inside-altadena-homes-survived-eaton-fire/15896312/",
    "https://www.cityofpasadena.net/city-manager/news/returning-home-after-a-fire-advisory-for-homes-impacted-by-eaton-fire/",
    "https://abc7.com/post/ask-abc7-could-toxic-ash-recent-fires-get-ground-soil-what-fruit-vegetable-trees/15847641/",
    "https://www.randrmagonline.com/articles/90793-cyanide-residues-in-fire-damaged-buildings",
    "https://www.providentfireplus.com/beware-hydrogen-cyanide-exposure/",
    "https://www.firehouse.com/rescue/article/10502165/hydrogen-cyanide-the-real-killer-among-fire-gases",
    "https://pubs.acs.org/doi/10.1021/acsestwater.1c00129",
    "https://publications.iarc.fr/_publications/media/download/6964/d95e533284e0053eb1cda28c5a11c442871dbfbe.pdf"
]

documents = []
for url in urls:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=" ")
        documents.append({"url": url, "content": text})
    except Exception as e:
        print(f"Error with {url}: {e}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for doc in documents:
    split_texts = splitter.split_text(doc["content"])
    for i, text in enumerate(split_texts):
        chunks.append({"url": doc["url"], "text": text, "id": f"{doc['url']}_{i}"})

with open("eaton_fire_docs.json", "w") as f:
    json.dump(chunks, f)

print("Knowledge base prepared and saved as eaton_fire_docs.json")
