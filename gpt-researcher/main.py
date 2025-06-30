from gpt_researcher import GPTResearcher
import asyncio

async def get_report(query: str, report_type: str, domains: list[str]):
    researcher = GPTResearcher(query, report_type, query_domains=domains)
    research_result = await researcher.conduct_research()
    report = await researcher.write_report()
    
    # Get additional information
    research_context = researcher.get_research_context()
    research_costs = researcher.get_costs()
    research_images = researcher.get_research_images()
    research_sources = researcher.get_research_sources()
    
    return report, research_context, research_costs, research_images, research_sources

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from a .env file (e.g., GOOGLE_API_KEY)

    report_type = "outline_report"
    domains = ['biopython.org', 'bioconductor.org']
    
    print("\n🚀 Toolfinder ready! Type 'quit' to exit. Type 'config' to configure your toolfinder.")
    while True:
        # Prompt the user to enter a query
        query = input("\nQuery: ").strip()
        if query.lower() in ["quit", "exit", "close"]:
            # Exit the loop if the user types 'quit'
            break
        if query.lower() == "config":
            while True:
                print('1) Report type: ' + report_type)
                print('exit')
                query = input("\n").strip()
                if query == "1":
                    print('1) research_report')
                    print('2) detailed_report')
                    print('3) deep')
                    print('4) resource_report')
                    print('5) outline_report (default)')
                    print('6) custom_report')
                    print('7) subtopic_report')

                    query = input("\n").strip()
                    if query.lower() in ["quit", "exit", "close"]:
                        # Exit the loop if the user types 'quit'
                        break
                    if query == "1":
                        report_type = "research_report"
                    if query == "2":
                        report_type = "detailed_report"
                    if query == "3":
                        report_type = "deep"
                    if query == "4":
                        report_type = "resource_report"
                    if query == "5":
                        report_type = "outline_report"
                    if query == "6":
                        report_type = "custom_report"
                    if query == "7":
                        report_type = "subtopic_report"
                if query.lower() in ["quit", "exit", "close"]:
                    # Exit the loop if the user types 'quit'
                    break
        if query.lower() in ["quit", "exit", "close"]:
            # Exit the loop if the user types 'quit'
            continue

        query += " \n Use biopython.org and bioconductor.org as references. Please add full links to the tools you found!"
    
        report, context, costs, images, sources = asyncio.run(get_report(query, report_type, domains))
    
        print("Report:")
        print(report)
        print("\nResearch Costs:")
        print(costs)
        print("\nNumber of Research Images:")
        print(len(images))
        print("\nNumber of Research Sources:")
        print(len(sources))