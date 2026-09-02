from agents.generation import generate_hypotheses
from agents.reflection import reflect_on_hypotheses
from agents.ranking import run_tournament
from agents.evolution import evolve_hypotheses
from tools.literature import search_literature
from agents.proximity import analyze_proximity
from agents.meta_review import create_research_roadmap

from core.state import ScientistState


class Supervisor:
    """Coordinates the multi-agent scientific reasoning process."""

    def __init__(
        self,
        research_goal: str,
        population_size: int = 3,
    ):
        self.state = ScientistState(
            research_goal=research_goal
        )

        self.population_size = population_size


    def initialize(self) -> None:
        """Generate the initial hypothesis population."""

        batch = generate_hypotheses(
            research_goal=self.state.research_goal,
            count=self.population_size,
            papers=self.state.papers,
        )

        self.state.hypotheses = batch.hypotheses

    def reflect(self) -> None:
        """Ask the reflection agent to review the population."""

        reviews = reflect_on_hypotheses(
            self.state.hypotheses
        )

        self.state.reviews.extend(reviews.reviews)

    def rank(self) -> None:
        """Run an Elo tournament."""

        self.state.hypotheses = run_tournament(
            self.state.hypotheses
        )

    def evolve(self) -> None:
        """Create a new generation from the strongest hypotheses."""

        strongest = self.state.hypotheses[:2]

        self.state.current_generation += 1

        evolved = evolve_hypotheses(
            hypotheses=strongest,
            count=self.population_size,
            generation=self.state.current_generation,
        )

        self.state.hypotheses = evolved.hypotheses

    def run(
        self,
        generations: int = 2,
        retrieve_literature: bool = True,
    ) -> ScientistState:
        """Execute the complete scientific reasoning loop."""

        if retrieve_literature:
            print("\n[Supervisor] Retrieving literature...")
            self.retrieve_literature()

            print(
                f"[Supervisor] Retrieved "
                f"{len(self.state.papers)} papers."
            )

        print(
            f"[Supervisor] Retrieved "
            f"{len(self.state.papers)} papers."
        )

        print("\n[Supervisor] Generating initial hypotheses...")
        self.initialize()

        for generation in range(generations):

            print(
                f"\n[Supervisor] Reflection "
                f"(generation {self.state.current_generation})..."
            )
            self.reflect()

            print(
                f"\n[Supervisor] Proximity analysis "
                f"(generation {self.state.current_generation})..."
            )
            self.analyze_similarity()

            print(
                f"\n[Supervisor] Ranking "
                f"(generation {self.state.current_generation})..."
            )
            self.rank()

            if generation < generations - 1:
                print(
                    f"\n[Supervisor] Evolving generation "
                    f"{self.state.current_generation}..."
                )
                self.evolve()

        print("\n[Supervisor] Creating final research roadmap...")
        self.meta_review()

        return self.state
    

    def retrieve_literature(self) -> None:
        """Retrieve scientific literature relevant to the research goal."""

        results = search_literature(
            query=self.state.research_goal,
            limit=5,
        )

        self.state.papers = results.papers

    def analyze_similarity(self) -> None:
        """Detect conceptually redundant hypotheses."""

        result = analyze_proximity(
            self.state.hypotheses
        )

        self.state.similarities.extend(
            result.pairs
        )

    def meta_review(self) -> None:
        """Create the final research roadmap."""

        self.state.roadmap = create_research_roadmap(
            research_goal=self.state.research_goal,
            hypotheses=self.state.hypotheses,
            reviews=self.state.reviews,
        )