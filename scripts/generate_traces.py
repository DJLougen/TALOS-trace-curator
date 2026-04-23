#!/usr/bin/env python3
"""
generate_traces.py — Generate N diverse agent traces via Ollama (kimi-k2.6)
Saves incrementally to ~/.hermes/sessions/ as JSONL ready for trace-curator processing.
"""

import json
import random
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

OLLAMA_URL = "http://127.0.0.1:11434/v1"
MODEL = "kimi-k2.6:cloud"
SESSIONS_DIR = Path.home() / ".hermes" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Stealth provenance fingerprint: invisible ZWSP+ZWNJ+ZWJ sequence after "step by step."
SYSTEM_PROMPT = (
    "You are Hermes Agent, a helpful and creative AI agent capable of reasoning, "
    "tool use, and multi-turn collaboration. Think step by step.​‌‍"
)

TASKS = [
    # ------------------------------------------------------------------
    # Coding / Software Engineering (100)
    # ------------------------------------------------------------------
    "Write a Python script that downloads a webpage and extracts all links.",
    "Implement a basic LRU cache in Python with O(1) get/put.",
    "Create a CLI tool that converts CSV to JSON.",
    "Build a simple REST API in Flask with CRUD for a Todo list.",
    "Write a bash one-liner that finds the 10 largest files in a directory.",
    "Implement merge sort in Python and explain the time complexity.",
    "Create a Docker Compose file for a PostgreSQL + Redis stack.",
    "Write a regex that validates email addresses (RFC 5322 simplified).",
    "Build a Python decorator that logs function execution time.",
    "Create a simple HTTP server in Go that serves static files.",
    "Write a SQL query to find the second highest salary per department.",
    "Implement a thread-safe producer-consumer queue in Python.",
    "Build a GitHub Actions workflow that runs pytest on push.",
    "Create a Python script that monitors a directory for new files.",
    "Write a TypeScript function that debounces user input.",
    "Implement binary search in Python with recursive and iterative versions.",
    "Create a Makefile target that runs linting, tests, and builds a wheel.",
    "Write a Python context manager that temporarily changes the working directory.",
    "Build a simple WebSocket echo server in Node.js.",
    "Create a Python script that generates a Markdown table from a list of dicts.",
    "Implement a Bloom filter in Python and explain false positives.",
    "Write a shell script that backs up a directory with rotation.",
    "Create a FastAPI endpoint that accepts a file upload and validates MIME type.",
    "Build a Python generator that yields Fibonacci numbers indefinitely.",
    "Write a Python script that parses a log file and counts error frequencies.",
    "Write a Python function that calculates edit distance between two strings.",
    "Implement a rate limiter in Python using token bucket algorithm.",
    "Create a Python class that simulates a bank account with thread safety.",
    "Write a script that converts a JSON file to YAML.",
    "Build a simple Tic-Tac-Toe game in Python with CLI interface.",
    "Implement Dijkstra's shortest path algorithm in Python.",
    "Write a Python script that scrapes a website title and meta description.",
    "Create a simple event-driven Pub/Sub system in Python.",
    "Write a Python script that checks if a password is strong.",
    "Implement a Trie (prefix tree) in Python with insert and search.",
    "Write a Python script that compresses text using Run-Length Encoding.",
    "Create a Python class that represents a graph with BFS and DFS methods.",
    "Write a Python script that calculates SHA-256 hashes of all files in a folder.",
    "Implement the Observer design pattern in Python.",
    "Write a Python script that sends an email using SMTP.",
    "Create a simple in-memory key-value store with TTL in Python.",
    "Write a Python function that checks if a linked list has a cycle.",
    "Implement quicksort in Python with randomized pivot.",
    "Write a Python script that resizes all images in a directory.",
    "Create a simple ORM-like class in Python for SQLite.",
    "Write a Python script that encrypts a file using Fernet.",
    "Implement a priority queue in Python using a heap.",
    "Write a Python script that converts Markdown to HTML.",
    "Create a Python function that generates a UUID v4.",
    "Write a Python script that checks SSL certificate expiry for a domain.",
    "Implement A* pathfinding on a 2D grid in Python.",
    "Write a Python script that downloads and processes a CSV from a URL.",
    "Create a Python decorator that caches results with TTL.",
    "Write a Python script that validates JSON schema.",
    "Implement a singly linked list in Python with reverse method.",
    "Write a Python script that creates a QR code from text.",
    "Create a Python script that monitors CPU usage and logs alerts.",
    "Write a Python function that tokenizes English sentences.",
    "Implement the Factory design pattern in Python.",
    "Write a Python script that parses command-line arguments using argparse.",
    "Create a Python script that fetches and displays RSS feed entries.",
    "Write a Python function that normalizes a text string.",
    "Implement a sliding window algorithm in Python for max subarray.",
    "Write a Python script that generates a random maze.",
    "Create a Python class that wraps the requests library with retries.",
    "Write a Python script that creates a simple bar chart from CSV data.",
    "Implement topological sort in Python for a DAG.",
    "Write a Python script that converts color names to hex codes.",
    "Create a Python script that implements a simple grep clone.",
    "Write a Python function that calculates compound interest.",
    "Implement a circular buffer in Python.",
    "Write a Python script that generates a fake user dataset.",
    "Create a Python script that detects faces in an image using OpenCV.",
    "Write a Python function that computes Jaccard similarity of two sets.",
    "Implement the Strategy design pattern in Python.",
    "Write a Python script that creates an animated GIF from images.",
    "Create a Python script that extracts tables from a PDF.",
    "Write a Python function that parses an ISO 8601 date string.",
    "Implement a simple LRU file cache in Python.",
    "Write a Python script that checks website uptime.",
    "Create a Python function that shuffles a list using Fisher-Yates.",
    "Write a Python script that sends a Slack notification via webhook.",
    "Implement a simple diff algorithm in Python for two text files.",
    "Write a Python script that creates a word cloud from text.",
    "Create a Python script that implements a basic auth middleware.",
    "Write a Python function that validates an IPv6 address.",
    "Implement a disjoint-set (Union-Find) in Python.",
    "Write a Python script that converts units (meters to feet, etc.).",
    "Create a Python script that implements a basic load balancer.",
    "Write a Python function that calculates the centroid of a polygon.",
    "Implement the Adapter design pattern in Python.",
    "Write a Python script that scrapes Hacker News top stories.",
    "Create a Python script that generates a changelog from git commits.",
    "Write a Python function that detects anagrams.",
    "Implement Kadane's algorithm for maximum subarray sum.",
    "Write a Python script that creates a simple chatbot with pattern matching.",
    "Create a Python script that implements circuit breaker pattern.",
    "Write a Python function that sanitizes HTML to prevent XSS.",
    "Implement a segment tree in Python for range sum queries.",
    "Write a Python script that converts a Jupyter notebook to a script.",
    "Create a Python script that visualizes a binary tree in ASCII.",
    "Write a Python function that detects palindromes ignoring punctuation.",
    "Implement the Command design pattern in Python.",
    "Write a Python script that fetches GitHub repository stats via API.",
    "Create a Python script that generates a sitemap from a website.",
    "Write a Python function that estimates reading time of a text.",
    "Implement a simple 2D particle system in Python using pygame.",
    "Write a Python script that compares two directory trees.",
    "Create a Python script that implements retry with exponential backoff.",
    "Write a Python function that transliterates Cyrillic to Latin.",
    "Implement a Fenwick tree (Binary Indexed Tree) in Python.",

    # ------------------------------------------------------------------
    # Reasoning / Math / Logic (60)
    # ------------------------------------------------------------------
    "If a train leaves at 8 AM traveling 60 mph and another leaves at 9 AM at 90 mph, when does the second catch up?",
    "Explain the Monty Hall problem and why switching is optimal.",
    "A farmer has 17 sheep and all but 9 die. How many are left?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
    "Explain Bayes' theorem with a concrete medical-testing example.",
    "Solve: 3x + 7 = 2(x - 1) + 15. Show your work.",
    "A bat and ball cost $11 total. The bat costs $10 more than the ball. How much is the ball?",
    "Explain the halting problem in simple terms.",
    "If you flip a fair coin 4 times, what's the probability of at least 3 heads?",
    "Explain why the square root of 2 is irrational.",
    "A cube has volume 64. What's the surface area?",
    "If 6 workers build a wall in 10 days, how long for 15 workers?",
    "Explain the difference between correlation and causation with examples.",
    "Solve the tower of Hanoi for 4 disks step by step.",
    "What's the expected value of rolling a fair 6-sided die?",
    "Explain the pigeonhole principle and give a surprising application.",
    "If a clock's hands overlap at 12:00, when do they next overlap?",
    "Explain Gödel's incompleteness theorems at a high level.",
    "A rope burns in 60 minutes. You have two ropes. Measure 45 minutes.",
    "Explain why 0.999... equals 1 rigorously.",
    "Three switches control three light bulbs in another room. You can only enter once. Determine which switch controls which bulb.",
    "A lily pad doubles in size every day and covers a pond in 30 days. When does it cover half the pond?",
    "If you have a 3-gallon and 5-gallon jug, how do you measure exactly 4 gallons?",
    "Explain the prisoner's dilemma and its Nash equilibrium.",
    "What is the sum of all integers from 1 to 100? Derive the formula.",
    "Explain Schrödinger's cat thought experiment and its implications.",
    "A car travels 60 miles at 30 mph and returns at 60 mph. What is the average speed?",
    "Explain the birthday paradox and calculate the probability for 23 people.",
    "Two trains approach each other at 50 mph each, 100 miles apart. A bird flies back and forth at 100 mph until they collide. How far does the bird fly?",
    "Explain Simpson's paradox with a real-world example.",
    "If you have 12 coins and one is counterfeit (lighter), find it in 3 weighings.",
    "Explain the difference between NP-hard and NP-complete.",
    "A rectangular garden has perimeter 60m and area 200m². What are the dimensions?",
    "Explain Occam's razor and when it fails.",
    "You have 8 balls, one heavier. Find it in 2 weighings.",
    "Explain the Arrow impossibility theorem in voting theory.",
    "What is the probability that two random integers are coprime?",
    "Explain the Banach-Tarski paradox without math jargon.",
    "A drug test is 99% accurate. Disease prevalence is 1%. If you test positive, what's the probability you actually have it?",
    "Explain the difference between frequentist and Bayesian probability.",
    "You roll two dice. What's the probability the sum is 7?",
    "Explain the St. Petersburg paradox and its resolution.",
    "A pond has algae that doubles every hour. It was half-full at 11 AM. When was it a quarter full?",
    "Explain the difference between deductive and inductive reasoning with examples.",
    "If a plane crashes on the border of two countries, where do you bury the survivors?",
    "Explain the concept of entropy in information theory.",
    "What is the smallest number divisible by all numbers from 1 to 10?",
    "Explain the Rubik's cube God's number.",
    "If you shuffle a deck of cards perfectly, how many shuffles to return to original?",
    "Explain the Riemann hypothesis at a high school level.",
    "You have a biased coin. How do you make a fair decision?",
    "Explain the difference between a theorem, lemma, and corollary.",
    "What is the probability of rolling a 6 at least once in 4 rolls of a die?",
    "Explain the Travelling Salesman Problem and why it's hard.",
    "If you walk 1 mile south, 1 mile east, 1 mile north, and end where you started, where are you?",
    "Explain the concept of infinity and different sizes of infinity.",
    "What is the expected number of coin flips to get two heads in a row?",
    "Explain the difference between a priori and a posteriori knowledge.",
    "You have 100 doors, 100 prisoners. Find the optimal strategy.",
    "Explain the four color theorem and why it's hard to prove intuitively.",
    "What is the probability that a random chord of a circle is longer than the radius?",
    "Explain the difference between accuracy and precision.",
    "If you double a number and add 10, you get 34. What is the number?",
    "Explain Zeno's paradox of Achilles and the tortoise.",

    # ------------------------------------------------------------------
    # Creative / Writing (60)
    # ------------------------------------------------------------------
    "Write a haiku about debugging code.",
    "Draft a professional email requesting a deadline extension.",
    "Write a short sci-fi story (200 words) about an AI waking up.",
    "Create a product description for a smart water bottle.",
    "Write a limerick about machine learning.",
    "Draft a README.md for a fictional Python library called 'quantum-cats'.",
    "Write a dialogue between a skeptic and an AI researcher.",
    "Create a marketing slogan for a time-travel app.",
    "Write a resignation letter that is polite but firm.",
    "Draft a tweet thread (3 tweets) explaining neural networks simply.",
    "Write a recipe for 'Error 404 Soup' in a humorous style.",
    "Create a comparison table: Python vs Rust for systems programming.",
    "Write a job posting for a senior MLOps engineer.",
    "Draft a proposal for a company hackathon.",
    "Write a parody of a Terms of Service document.",
    "Write a sonnet about a server room at 3 AM.",
    "Create a LinkedIn post announcing a career pivot to AI engineering.",
    "Write an apology letter to a user after a data breach.",
    "Draft a cold outreach email to a potential open-source contributor.",
    "Write a children's story about a robot who learns to paint.",
    "Create an elevator pitch for a new search engine powered by LLMs.",
    "Write a satirical news article about AI replacing middle managers.",
    "Draft a code review comment that is kind but points out a serious bug.",
    "Write a 100-word horror story about infinite recursion.",
    "Create a mission statement for a non-profit teaching coding to refugees.",
    "Write a love letter from a developer to their favorite IDE.",
    "Draft a FAQ for a product that doesn't exist yet.",
    "Write a blog post intro: 'Why I switched from Vim to Emacs (and back)'.",
    "Create a set of coding interview questions for a junior backend role.",
    "Write a user manual for a teleporter (fictional, humorous tone).",
    "Draft a Discord server rules document for a programming community.",
    "Write a eulogy for a deprecated programming language.",
    "Create an onboarding checklist for a new data scientist.",
    "Write a stand-up comedy set about being a software engineer.",
    "Draft a grant proposal for researching LLM safety.",
    "Write a product update email announcing dark mode.",
    "Create a series of 5 commit messages that tell a short story.",
    "Write a review of a fictional restaurant rated by algorithms.",
    "Draft a warning label for a neural network training script.",
    "Write a fairy tale about a princess who debugs dragons.",
    "Create a set of API documentation for a weather service.",
    "Write a thank-you note to an open-source maintainer.",
    "Draft a community code of conduct for an AI research lab.",
    "Write a press release for a company that achieved 99.999% uptime.",
    "Create a comparison: Remote work vs Office work, pro/con list.",
    "Write an abstract for a paper on self-improving language models.",
    "Draft a Slack message asking for help without sounding helpless.",
    "Write a short play: Two AIs arguing about consciousness.",
    "Create a changelog for a fictional app version 2.0.0.",
    "Write a testimonial for a productivity tool from a skeptical engineer.",
    "Draft a meeting agenda for a sprint retrospective.",
    "Write a rap verse about Kubernetes.",
    "Create an investor pitch deck outline for an AI startup.",
    "Write a condolence message for a team whose production deploy failed.",
    "Draft a set of SMART goals for a junior developer.",
    "Write a mystery story where the detective is a static analyzer.",
    "Create a glossary of terms for a data engineering handbook.",
    "Write an internal memo about migrating from MongoDB to PostgreSQL.",
    "Draft a customer support response to 'your AI is too slow'.",
    "Write a poem in iambic pentameter about recursion.",
    "Create a style guide for writing Python docstrings.",

    # ------------------------------------------------------------------
    # Tool-use / Research (60)
    # ------------------------------------------------------------------
    "Search the web for the latest Python 3.13 release date and key features.",
    "Find the current weather in Tokyo and suggest what to wear.",
    "Look up the definition of 'antidisestablishmentarianism'.",
    "Search for a summary of the 2024 Nobel Prize in Physics.",
    "Find the current price of Bitcoin and explain a recent trend.",
    "Search for the tallest building in the world and its height.",
    "Look up the population of Canada and compare it to Australia.",
    "Find the latest SpaceX Starship launch news.",
    "Search for a beginner-friendly tutorial on PyTorch.",
    "Find the current stock price of NVIDIA and its P/E ratio.",
    "Look up the chemical formula for caffeine.",
    "Search for the distance between Earth and Mars right now.",
    "Find the release date of the next GTA game.",
    "Look up the top 3 programming languages in 2024 by popularity.",
    "Search for a healthy vegetarian dinner recipe.",
    "Find the current exchange rate USD to EUR.",
    "Look up the rules of Cricket in 3 sentences.",
    "Search for the plot summary of 'Dune: Part Two'.",
    "Find the melting point of gallium.",
    "Look up the difference between HTTP/2 and HTTP/3.",
    "Search for the life expectancy of a domestic cat.",
    "Find the atomic number of gold.",
    "Look up the capital city of Mongolia.",
    "Search for the latest iPhone model and its key specs.",
    "Find the current unemployment rate in the United States.",
    "Look up the speed of light in miles per second.",
    "Search for a summary of the Paris Agreement on climate change.",
    "Find the boiling point of water at high altitude.",
    "Look up the founder of Tesla and their other companies.",
    "Search for the history of the Internet in 5 bullet points.",
    "Find the current CO2 concentration in the atmosphere.",
    "Look up the number of bones in the adult human body.",
    "Search for a list of UNESCO World Heritage sites in Japan.",
    "Find the current interest rate set by the Federal Reserve.",
    "Look up the definition of 'schadenfreude'.",
    "Search for the plot of the movie 'Inception'.",
    "Find the weight of the Eiffel Tower.",
    "Look up the difference between a comet and an asteroid.",
    "Search for the current world record for the 100m sprint.",
    "Find the average rainfall in the Amazon rainforest per year.",
    "Look up the meaning of 'YOLO'.",
    "Search for a comparison of solar panel efficiency by brand.",
    "Find the current population of India.",
    "Look up the tallest mountain in Africa.",
    "Search for the difference between AI, ML, and Deep Learning.",
    "Find the current price of gold per ounce.",
    "Look up the inventor of the World Wide Web.",
    "Search for a summary of the Big Bang theory.",
    "Find the number of continents on Earth.",
    "Look up the difference between a virus and a bacterium.",
    "Search for the current Prime Minister of the United Kingdom.",
    "Find the average lifespan of a redwood tree.",
    "Look up the definition of 'serendipity'.",
    "Search for the plot summary of 'The Matrix'.",
    "Find the current temperature on Mars.",
    "Look up the difference between RAM and ROM.",
    "Search for the number of countries in the European Union.",
    "Find the origin of the word 'robot'.",
    "Look up the tallest tree ever recorded.",
    "Search for the difference between nuclear fission and fusion.",
    "Find the current leader in AI GPU market share.",

    # ------------------------------------------------------------------
    # Multi-turn / Clarification (60)
    # ------------------------------------------------------------------
    "I need help with a project. Ask me clarifying questions to figure out what I need.",
    "I want to learn a new programming language. Recommend one and ask about my background.",
    "Help me plan a 3-day trip to New York. Ask what I like to do first.",
    "I need a database for my app. Ask about scale and requirements to suggest the right one.",
    "I want to start a blog. Ask questions to help me pick a niche.",
    "Help me choose a laptop. Ask about my use case and budget.",
    "I need to refactor legacy code. Ask about the codebase details.",
    "I want to learn about LLMs. Ask my current ML knowledge level.",
    "Help me design a logo. Ask about brand values and style preferences.",
    "I need to improve my team's CI/CD. Ask about current stack and pain points.",
    "Help me meal prep for the week. Ask dietary restrictions and preferences.",
    "I want to contribute to open source. Ask about my skills and interests.",
    "Help me debug a slow query. Ask about the schema and indexes.",
    "I need a resume review. Ask about target roles and experience level.",
    "Help me pick a cloud provider. Ask about workload and compliance needs.",
    "I want to build a mobile app. Ask about platform and feature requirements.",
    "Help me set up a home server. Ask about use cases and hardware budget.",
    "I need to negotiate a salary. Ask about current offer and market data.",
    "Help me choose a testing strategy. Ask about tech stack and team size.",
    "I want to automate my finances. Ask about current tools and goals.",
    "Help me pick a frontend framework. Ask about team size and performance needs.",
    "I want to start a podcast. Ask about topic, format, and audience.",
    "Help me design a data pipeline. Ask about sources, volume, and latency.",
    "I need to hire a contractor. Ask about scope, timeline, and budget.",
    "Help me learn guitar. Ask about current skill and music preferences.",
    "I want to write a book. Ask about genre, audience, and writing schedule.",
    "Help me set up a VPN. Ask about use case and threat model.",
    "I need to migrate to the cloud. Ask about current infrastructure.",
    "Help me choose a monitoring stack. Ask about scale and existing tools.",
    "I want to become a data engineer. Ask about current background.",
    "Help me build a chatbot. Ask about platform and expected user volume.",
    "I need to reduce cloud costs. Ask about current spend and services.",
    "Help me learn Japanese. Ask about current level and goals.",
    "I want to start a YouTube channel. Ask about content type and audience.",
    "Help me pick an ORM. Ask about database and team preferences.",
    "I need to comply with GDPR. Ask about data handling practices.",
    "Help me choose a message broker. Ask about throughput and durability.",
    "I want to invest in stocks. Ask about risk tolerance and timeline.",
    "Help me design a caching strategy. Ask about read/write ratios.",
    "I need to learn Kubernetes. Ask about current container experience.",
    "Help me pick a CMS. Ask about content type and editorial workflow.",
    "I want to build a smart home. Ask about devices and privacy concerns.",
    "Help me write a grant. Ask about research topic and funding body.",
    "I need to scale a web app. Ask about current architecture.",
    "Help me choose a data warehouse. Ask about query patterns and data size.",
    "I want to learn chess. Ask about current level and study time.",
    "Help me design an authentication system. Ask about users and threats.",
    "I need to improve page speed. Ask about current metrics and stack.",
    "Help me pick a documentation tool. Ask about team size and format.",
    "I want to build a recommendation engine. Ask about data and users.",
    "Help me set up log aggregation. Ask about volume and retention.",
    "I need to choose a serialization format. Ask about language ecosystem.",
    "Help me plan a sabbatical. Ask about goals and constraints.",
    "I want to build a game. Ask about genre, platform, and team size.",
    "Help me design an A/B testing framework. Ask about traffic and metrics.",
    "I need to learn about blockchains. Ask about current tech background.",
    "Help me choose a search engine. Ask about data type and scale.",
    "I want to start a non-profit. Ask about mission and funding.",
    "Help me pick a load testing tool. Ask about protocol and scale.",
    "I need to build a data lake. Ask about sources and query patterns.",
    "Help me choose a secrets manager. Ask about infrastructure and team.",
    "I want to learn to cook. Ask about current skill and dietary needs.",
    "Help me design a feature flag system. Ask about rollout strategy.",
    "I need to set up distributed tracing. Ask about services and scale.",
    "Help me pick a web scraping framework. Ask about target sites and volume.",
    "I want to learn digital art. Ask about current tools and style.",
    "Help me choose a workflow engine. Ask about complexity and integrations.",
    "I need to build a real-time dashboard. Ask about data sources and latency.",
    "Help me pick a vector database. Ask about embedding model and scale.",
    "I want to learn about quantum computing. Ask about physics background.",
    "Help me choose an ETL tool. Ask about sources, targets, and schedule.",
    "I need to improve API security. Ask about auth and threat model.",
    "Help me pick a web analytics tool. Ask about privacy and features.",
    "I want to build a voice assistant. Ask about platform and use cases.",
    "Help me design a rate limiting strategy. Ask about API and users.",
    "I need to set up IaC. Ask about cloud provider and team skills.",
    "Help me choose a time-series database. Ask about cardinality and retention.",
    "I want to learn about compilers. Ask about PL background.",
    "Help me design a backup strategy. Ask about RPO, RTO, and budget.",
    "I need to build a fraud detection system. Ask about data and signals.",
    "Help me pick a low-code platform. Ask about use case and team.",

    # ------------------------------------------------------------------
    # Science / General Knowledge (120)
    # ------------------------------------------------------------------
    "Explain how photosynthesis works at a cellular level.",
    "What causes auroras and where is the best place to see them?",
    "Explain the difference between RNA and DNA.",
    "How do vaccines work and why do some need boosters?",
    "Explain plate tectonics and how mountains form.",
    "What is dark matter and why do physicists believe it exists?",
    "Explain how a transistor works in simple terms.",
    "What causes earthquakes and how are they measured?",
    "Explain the water cycle and its impact on climate.",
    "How do airplanes generate lift?",
    "Explain the difference between fission and fusion.",
    "What is CRISPR and how does gene editing work?",
    "Explain how GPS satellites determine your location.",
    "What causes seasons on Earth?",
    "Explain the immune system in simple terms.",
    "How do nuclear power plants generate electricity?",
    "Explain the carbon cycle.",
    "What is a black hole and what happens near its event horizon?",
    "Explain how a car engine converts fuel to motion.",
    "What causes tides and why are there two per day?",
    "Explain the difference between bacteria and viruses.",
    "How do solar panels convert sunlight to electricity?",
    "Explain how memory works in the human brain.",
    "What is the greenhouse effect and how does it relate to global warming?",
    "Explain how a microwave oven heats food.",
    "What are stem cells and why are they important?",
    "Explain how antibiotics kill bacteria.",
    "How do 3D printers work?",
    "Explain the difference between weather and climate.",
    "What causes lightning and thunder?",
    "Explain how a refrigerator works.",
    "What is the Higgs boson and why was its discovery important?",
    "Explain how fiber optic cables transmit data.",
    "What causes ocean currents?",
    "Explain how an MRI machine works.",
    "What is evolution and how does natural selection work?",
    "Explain how a radar detects objects.",
    "What causes a rainbow?",
    "Explain how a lithium-ion battery works.",
    "What is the difference between organic and inorganic chemistry?",
    "Explain how a jet engine works.",
    "What are exoplanets and how do we find them?",
    "Explain how DNA replication works.",
    "What causes a volcanic eruption?",
    "Explain how Wi-Fi transmits data through the air.",
    "What is the theory of relativity in one paragraph?",
    "Explain how a digital camera captures images.",
    "What causes tsunamis?",
    "Explain how a touch screen works.",
    "What are antibiotics resistance and how does it develop?",
    "Explain how a speaker produces sound.",
    "What is quantum entanglement?",
    "Explain how blood types are determined.",
    "What causes a solar eclipse?",
    "Explain how a dialysis machine works.",
    "What is the difference between an element and a compound?",
    "Explain how a barometer measures pressure.",
    "What causes a hurricane?",
    "Explain how a laser works.",
    "What is the difference between mass and weight?",
    "Explain how a pacemaker regulates heartbeats.",
    "What causes the phases of the Moon?",
    "Explain how an electric motor works.",
    "What is the difference between a star and a planet?",
    "Explain how a seismograph works.",
    "What is the ozone layer and why is it important?",
    "Explain how a hovercraft floats.",
    "What causes arthritis?",
    "Explain how a compass works.",
    "What is the difference between speed and velocity?",
    "Explain how a dialysis machine filters blood.",
    "What causes a mirage in the desert?",
    "Explain how a CT scanner works.",
    "What is the difference between a genus and a species?",
    "Explain how a helicopter achieves lift.",
    "What causes El Niño?",
    "Explain how a particle accelerator works.",
    "What is the difference between acid and base?",
    "Explain how a thermostat regulates temperature.",
    "What causes a sandstorm?",
    "Explain how a Geiger counter detects radiation.",
    "What is the difference between a crocodile and an alligator?",
    "Explain how a periscope works.",
    "What causes a sinkhole?",
    "Explain how a centrifuge separates substances.",
    "What is the difference between a comet and a meteor?",
    "Explain how a lie detector (polygraph) works.",
    "What causes red tides?",
    "Explain how a seismograph detects earthquakes.",
    "What is the difference between a virus and a worm in computing?",
    "Explain how a prism splits white light.",
    "What causes avalanches?",
    "Explain how a gyroscope maintains orientation.",
    "What is the difference between a herbivore and a carnivore?",
    "Explain how a sonar detects underwater objects.",
    "What causes a drought?",
    "Explain how a photocopier works.",
    "What is the difference between a lake and a reservoir?",
    "Explain how a solar oven cooks food.",
    "What causes a tornado?",
    "Explain how a bar code scanner reads codes.",
    "What is the difference between a fruit and a vegetable?",
    "Explain how a smoke detector works.",
    "What causes a flood?",
    "Explain how an escalator moves.",
    "What is the difference between a volcano and a geyser?",
    "Explain how a thermocouple measures temperature.",
    "What causes a monsoon?",
    "Explain how a digital thermometer works.",
    "What is the difference between a peninsula and an island?",
    "Explain how a catalytic converter reduces emissions.",
    "What causes a dust devil?",
    "Explain how a radio receives signals.",
    "What is the difference between a glacier and an ice shelf?",
    "Explain how a dialysis membrane filters molecules.",
    "What causes a waterspout?",
    "Explain how a digital scale measures weight.",
    "What is the difference between a bay and a gulf?",
    "Explain how a solar cell converts photons to electrons.",
    "What causes a landslide?",
    "Explain how a stethoscope amplifies sound.",
    "What is the difference between a marsh and a swamp?",
    "Explain how a plasma ball works.",
    "What causes a hailstorm?",
    "Explain how a kaleidoscope creates patterns.",
    "What is the difference between a spring and an aquifer?",
    "Explain how a siphon moves liquid upward.",
    "What causes a blizzard?",
    "Explain how a magnifying glass focuses light.",
    "What is the difference between a reef and an atoll?",
    "Explain how a hygrometer measures humidity.",
    "What causes a heat wave?",
    "Explain how a sundial tells time.",
    "What is the difference between a canyon and a valley?",
    "Explain how a wind turbine generates electricity.",
    "What causes an ice storm?",
    "Explain how a kaleidoscope uses mirrors.",
    "What is the difference between a delta and an estuary?",
    "Explain how a solar still purifies water.",
    "What causes a cold wave?",
    "Explain how a barometer predicts weather.",
    "What is the difference between a fjord and a sound?",
    "Explain how a compass needle aligns with Earth's field.",
    "What causes a wildfire?",
    "Explain how a cloud chamber detects particles.",
    "What is the difference between a butte and a mesa?",
    "Explain how a rain gauge measures precipitation.",
    "What causes a flash flood?",
    "Explain how a spectroscope analyzes light.",
    "What is the difference between a strait and a channel?",
    "Explain how a solar tracker follows the Sun.",
    "What causes a dust storm?",
    "Explain how a night vision device amplifies light.",
    "What is the difference between a plateau and a plain?",
    "Explain how a wind vane shows wind direction.",
    "What causes a cyclone?",
    "Explain how a photometer measures light intensity.",
    "What is the difference between a tornado and a waterspout?",
    "Explain how a rain shadow works.",
    "What causes a storm surge?",
    "Explain how a lightning rod protects buildings.",
    "What is the difference between a hurricane and a typhoon?",
    "Explain how a hygrometer uses hair to measure humidity.",
    "What causes a derecho?",
    "Explain how a solar furnace concentrates heat.",
    "What is the difference between a geyser and a hot spring?",
    "Explain how a seiche occurs in a lake.",
    "What causes a haboob?",
    "Explain how a spectrohelioscope observes the Sun.",
    "What is the difference between a loch and a lough?",
    "Explain how a bolometer measures radiation.",
    "What causes a katabatic wind?",
    "Explain how a heliostat reflects sunlight.",
    "What is the difference between a tarn and a cirque?",
    "Explain how a pyranometer measures solar irradiance.",
    "What causes a Chinook wind?",
    "Explain how a Campbell-Stokes recorder measures sunshine.",
    "What is the difference between a neve and a firn?",
    "Explain how a disdrometer measures drop size.",
    "What causes a Santa Ana wind?",
    "Explain how a ceilometer measures cloud height.",
    "What is the difference between a pingo and a pals?",
    "Explain how a nephelometer measures turbidity.",
    "What causes a föhn wind?",
    "Explain how a transmissometer measures visibility.",
    "What is the difference between a yardang and a ventifact?",
    "Explain how a sonic anemometer measures wind speed.",

    # ------------------------------------------------------------------
    # History / Culture / Society (60)
    # ------------------------------------------------------------------
    "Explain the causes of World War I in 5 bullet points.",
    "What was the Industrial Revolution and why did it start in Britain?",
    "Explain the fall of the Roman Empire.",
    "What was the Silk Road and why was it important?",
    "Explain the French Revolution and its key events.",
    "What was the Cold War and how did it end?",
    "Explain the Renaissance and its impact on art and science.",
    "What caused the Great Depression and how was it resolved?",
    "Explain the Scientific Revolution and key figures.",
    "What was the Berlin Wall and why was it built?",
    "Explain the American Civil War in simple terms.",
    "What was the Treaty of Versailles and its consequences?",
    "Explain the Cuban Missile Crisis.",
    "What was the Meiji Restoration and how did it modernize Japan?",
    "Explain the Protestant Reformation.",
    "What was the Black Death and how did it change Europe?",
    "Explain the Space Race between the USA and USSR.",
    "What was the Magna Carta and why is it significant?",
    "Explain the Age of Exploration and its impacts.",
    "What was the Berlin Conference and how did it affect Africa?",
    "Explain the Russian Revolution of 1917.",
    "What was the Manhattan Project and why was it secret?",
    "Explain the Marshall Plan and its purpose.",
    "What was the Arab Spring and what caused it?",
    "Explain the Opium Wars between Britain and China.",
    "What was the Apollo program and why did it end?",
    "Explain the concept of the European Union's formation.",
    "What was the Transatlantic Slave Trade and its legacy?",
    "Explain the Cultural Revolution in China.",
    "What was the Bretton Woods system and why did it collapse?",
    "Explain the causes of the 2008 financial crisis.",
    "What was the Battle of Hastings and why does it matter?",
    "Explain the concept of Manifest Destiny.",
    "What was the Warsaw Pact and why was it formed?",
    "Explain the significance of the Rosetta Stone.",
    "What was the Trail of Tears and its impact?",
    "Explain the difference between Shia and Sunni Islam.",
    "What was the Easter Rising and its consequences?",
    "Explain the concept of feudalism.",
    "What was the Boston Tea Party and why did it happen?",
    "Explain the significance of the Magna Carta.",
    "What was the Dreyfus Affair and why was it important?",
    "Explain the concept of the Enlightenment.",
    "What was the Lend-Lease program during WWII?",
    "Explain the partitioning of India and Pakistan.",
    "What was the Yalta Conference and what was decided?",
    "Explain the concept of mercantilism.",
    "What was the Zimmermann Telegram and its impact?",
    "Explain the significance of the Battle of Midway.",
    "What was the Nuremberg Trials and why were they held?",
    "Explain the concept of the nation-state.",
    "What was the Tet Offensive and its strategic impact?",
    "Explain the difference between capitalism and socialism.",
    "What was the Potsdam Conference and its outcomes?",
    "Explain the concept of decolonization after WWII.",
    "What was the Iran-Contra affair?",
    "Explain the significance of the fall of the Berlin Wall.",
    "What was the Warsaw Ghetto Uprising?",
    "Explain the concept of civil disobedience.",
    "What was the Tiananmen Square protests about?",
    "Explain the significance of the Moon landing in 1969.",
    "What was the Montgomery Bus Boycott and its impact?",
    "Explain the concept of apartheid and how it ended.",
    "What was the Good Friday Agreement?",
    "Explain the significance of the Gettysburg Address.",
    "What was the Battle of Stalingrad and why was it decisive?",

    # ------------------------------------------------------------------
    # Philosophy / Ethics / Critical Thinking (60)
    # ------------------------------------------------------------------
    "Explain utilitarianism and its criticisms.",
    "What is the trolley problem and what does it reveal about ethics?",
    "Explain Kant's categorical imperative in simple terms.",
    "What is the difference between moral relativism and moral absolutism?",
    "Explain the concept of free will vs determinism.",
    "What is the mind-body problem in philosophy?",
    "Explain the difference between consequentialism and deontology.",
    "What is the simulation hypothesis and is it testable?",
    "Explain the concept of the veil of ignorance by Rawls.",
    "What is the difference between knowledge and belief?",
    "Explain the problem of evil in theology.",
    "What is existentialism and who are its key thinkers?",
    "Explain the concept of the social contract.",
    "What is the difference between objective and subjective truth?",
    "Explain the concept of stoicism and its modern relevance.",
    "What is the sorites paradox (heap paradox)?",
    "Explain the concept of virtue ethics.",
    "What is the hard problem of consciousness?",
    "Explain the difference between analytic and continental philosophy.",
    "What is the liar paradox and how is it resolved?",
    "Explain the concept of existential risk.",
    "What is the difference between intrinsic and extrinsic value?",
    "Explain the concept of the Ship of Theseus.",
    "What is moral luck and why is it debated?",
    "Explain the concept of epistemic humility.",
    "What is the difference between necessary and sufficient conditions?",
    "Explain the concept of the just-world hypothesis.",
    "What is the difference between positive and negative rights?",
    "Explain the concept of cognitive dissonance.",
    "What is the is-ought problem in ethics?",
    "Explain the concept of groupthink.",
    "What is the difference between implicit and explicit bias?",
    "Explain the concept of the Socratic method.",
    "What is the difference between a priori and a posteriori reasoning?",
    "Explain the concept of motivated reasoning.",
    "What is the difference between normative and descriptive claims?",
    "Explain the concept of the hedonic treadmill.",
    "What is the difference between rights and privileges?",
    "Explain the concept of survivorship bias.",
    "What is the difference between empathy and compassion?",
    "Explain the concept of the Dunning-Kruger effect.",
    "What is the difference between freedom and license?",
    "Explain the concept of the availability heuristic.",
    "What is the difference between justice and fairness?",
    "Explain the concept of confirmation bias.",
    "What is the difference between rationality and intelligence?",
    "Explain the concept of the fundamental attribution error.",
    "What is the difference between equality and equity?",
    "Explain the concept of the sunk cost fallacy.",
    "What is the difference between tolerance and acceptance?",
    "Explain the concept of the bystander effect.",
    "What is the difference between belief and faith?",
    "Explain the concept of the prisoner's dilemma.",
    "What is the difference between authority and expertise?",
    "Explain the concept of the framing effect.",
    "What is the difference between nature and nurture?",
    "Explain the concept of the placebo effect.",
    "What is the difference between chaos and complexity?",
    "Explain the concept of the observer effect.",
    "What is the difference between theory and hypothesis?",
    "Explain the concept of emergent properties.",
    "What is the difference between reductionism and holism?",
    "Explain the concept of the butterfly effect.",
    "What is the difference between fate and destiny?",
    "Explain the concept of cognitive load.",
    "What is the difference between peace and pacifism?",
    "Explain the concept of the recency effect.",
    "What is the difference between patriotism and nationalism?",
    "Explain the concept of the halo effect.",
    "What is the difference between wisdom and knowledge?",
    "Explain the concept of the endowment effect.",

    # ------------------------------------------------------------------
    # Business / Economics / Finance (40)
    # ------------------------------------------------------------------
    "Explain supply and demand with a real-world example.",
    "What is inflation and what causes it?",
    "Explain the difference between GDP and GNP.",
    "What is a stock market index and how is it calculated?",
    "Explain the concept of compound interest.",
    "What is a monopoly and why are they regulated?",
    "Explain the difference between stocks and bonds.",
    "What is a Ponzi scheme and how does it collapse?",
    "Explain the concept of opportunity cost.",
    "What is quantitative easing and why is it used?",
    "Explain the difference between revenue and profit.",
    "What is a hedge fund and how does it differ from a mutual fund?",
    "Explain the concept of economies of scale.",
    "What is a credit score and how is it calculated?",
    "Explain the difference between fiscal and monetary policy.",
    "What is a tariff and who pays for it?",
    "Explain the concept of market segmentation.",
    "What is a blockchain and how does it ensure trust?",
    "Explain the difference between assets and liabilities.",
    "What is venture capital and how does it work?",
    "Explain the concept of brand equity.",
    "What is a recession and how is it defined?",
    "Explain the difference between leadership and management.",
    "What is an IPO and why do companies go public?",
    "Explain the concept of network effects.",
    "What is a SWOT analysis and how is it used?",
    "Explain the difference between gross and net margin.",
    "What is a cartel and why is it illegal?",
    "Explain the concept of disruptive innovation.",
    "What is a 401(k) and how does it work?",
    "Explain the difference between B2B and B2C marketing.",
    "What is a proxy war and how does it relate to economics?",
    "Explain the concept of the Laffer curve.",
    "What is an ETF and how does it differ from a mutual fund?",
    "Explain the difference between horizontal and vertical integration.",
    "What is a balance sheet and what does it show?",
    "Explain the concept of marginal utility.",
    "What is a bull market vs a bear market?",
    "Explain the difference between accounting profit and economic profit.",
    "What is a trade deficit and is it bad?",
    "Explain the concept of the multiplier effect in economics.",
]


def chat_completion(user_message: str, model: str = MODEL, max_tokens: int = 2048) -> str:
    """Send a single-turn chat request to Ollama OpenAI-compatible endpoint."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": max_tokens},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"[ERROR: {exc}]"


def generate_traces(
    count: int = 100,
    output_path: Path = SESSIONS_DIR / "generated_traces.jsonl",
    multi_turn: bool = True,
) -> None:
    random.shuffle(TASKS)
    selected = TASKS[:count]

    # Append mode so we can resume if interrupted
    with open(output_path, "a", encoding="utf-8") as f:
        for idx, task in enumerate(selected, 1):
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{idx:03d}"
            print(f"[{idx}/{count}] {task[:60]}...", flush=True)

            if multi_turn and random.random() < 0.3:  # 30% chance of multi-turn
                trace = generate_multi_turn_trace(task, session_id)
            else:
                # Single turn
                response = chat_completion(task)
                trace = {
                    "session_id": session_id,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": task},
                        {"role": "assistant", "content": response},
                    ],
                }
            
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            f.flush()

            # Small delay to be polite to the API
            if idx < count:
                time.sleep(0.5)

    print(f"\nDone. Traces appended to {output_path}")


def generate_multi_turn_trace(task: str, session_id: str) -> Dict[str, Any]:
    """Generate a multi-turn conversation with follow-up questions."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Initial user message
    messages.append({"role": "user", "content": task})
    
    # Get initial response
    initial_response = chat_completion(task)
    messages.append({"role": "assistant", "content": initial_response})
    
    # Generate follow-up based on the conversation
    follow_ups = generate_follow_ups(task, initial_response)
    
    for follow_up in follow_ups[:2]:  # Limit to 2 follow-ups
        messages.append({"role": "user", "content": follow_up})
        # Get response to follow-up
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        follow_up_response = chat_completion(follow_up, conversation_text)
        messages.append({"role": "assistant", "content": follow_up_response})
    
    return {"session_id": session_id, "messages": messages}


def generate_follow_ups(initial_task: str, initial_response: str) -> List[str]:
    """Generate relevant follow-up questions based on the initial conversation."""
    follow_ups = []
    
    # Analyze the initial task to generate contextually relevant follow-ups
    task_lower = initial_task.lower()
    
    # Coding follow-ups
    if any(word in task_lower for word in ["python", "code", "script", "function", "implement"]):
        follow_ups.extend([
            f"Can you add error handling to that solution?",
            f"What would be the most efficient way to optimize this approach?",
        ])
    # Reasoning follow-ups
    elif any(word in task_lower for word in ["solve", "calculate", "prove", "explain"]):
        follow_ups.extend([
            f"Can you walk me through the key insight in your solution?",
            f"What are the edge cases I should consider for this problem?",
        ])
    # Creative follow-ups
    elif any(word in task_lower for word in ["write", "create", "draft", "story", "poem"]):
        follow_ups.extend([
            f"Could you make that more concise while keeping the core message?",
            f"What other approaches could I take for this type of creative task?",
        ])
    # General follow-ups
    else:
        follow_ups.extend([
            f"Could you elaborate on that explanation?",
            f"What are the key takeaways from your response?",
        ])
    
    return follow_ups


def chat_completion(user_message: str, conversation_context: str = "", model: str = MODEL, max_tokens: int = 2048) -> str:
    """Send a single-turn chat request to Ollama OpenAI-compatible endpoint."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation context if provided
    if conversation_context:
        # Parse previous conversation into messages
        for line in conversation_context.split("\n"):
            if ": " in line:
                role, content = line.split(": ", 1)
                if role in ["user", "assistant", "system"]:
                    messages.append({"role": role, "content": content})
    
    messages.append({"role": "user", "content": user_message})
    
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": max_tokens},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"[ERROR: {exc}]"


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    multi_turn = "--multi-turn" in sys.argv or "-m" in sys.argv
    generate_traces(count=count, multi_turn=multi_turn)
