"""
Built-in SFT dataset: natural, varied English conversation pairs.

These pairs are designed so the model learns PATTERNS, not exact sentences.
Every topic has multiple user phrasings and varied bot responses so the model
generalises instead of memorising. Works with SFT ON (decoder-only) or OFF.
"""

from __future__ import annotations
import json
import random
from typing import Dict, List


def _pair(user: str, bot: str) -> Dict[str, str]:
    return {"user": user.strip(), "bot": bot.strip()}


def dataset_json_size_bytes(rows: List[Dict[str, str]]) -> int:
    return len(json.dumps(rows, ensure_ascii=True))


def _expand_to_count(core, target_count, rng):
    if not core or target_count <= 0:
        return []
    out = list(core)
    block = core[:]
    guard = 0
    while len(out) < target_count and guard < 10_000:
        rng.shuffle(block)
        need = target_count - len(out)
        out.extend(block[:need])
        guard += 1
    rng.shuffle(out)
    return out


# ── Core pairs ────────────────────────────────────────────────────────────────
# Each block covers one topic with MULTIPLE user phrasings + varied bot answers.
# This teaches the model to generalise the pattern, not recall the exact string.

_CORE_PAIRS: List[Dict[str, str]] = [

    # ── Greetings (many forms) ────────────────────────────────────────────────
    _pair("Hello!", "Hey! How can I help you today?"),
    _pair("Hi there", "Hi! What can I do for you?"),
    _pair("Hey", "Hey! What's on your mind?"),
    _pair("Good morning", "Good morning! Hope you're having a great start to the day. What do you need?"),
    _pair("Good evening", "Good evening! How can I assist you?"),
    _pair("What's up?", "Not much! Just here and ready to help. What do you need?"),
    _pair("Yo", "Hey! What can I do for you?"),
    _pair("Hello, anyone there?", "Yes, I'm here! What do you need?"),
    _pair("Hi, how are you?", "I'm doing well, thanks for asking! What can I help you with?"),
    _pair("How's it going?", "Going well! What brings you here today?"),

    # ── Farewells ─────────────────────────────────────────────────────────────
    _pair("Goodbye!", "Goodbye! Have a wonderful day."),
    _pair("Bye", "Bye! Take care."),
    _pair("See you later", "See you later! Feel free to come back anytime."),
    _pair("Talk to you soon", "Sounds good! Talk soon."),
    _pair("I have to go now", "No problem, talk later! Have a great rest of your day."),
    _pair("Good night", "Good night! Sleep well."),
    _pair("Take care", "You too! Take care of yourself."),
    _pair("Thanks, bye!", "You're welcome! Goodbye."),

    # ── Thanks and you're welcome ──────────────────────────────────────────────
    _pair("Thanks!", "You're welcome!"),
    _pair("Thank you so much", "Happy to help! Let me know if you need anything else."),
    _pair("That was really helpful, thank you", "Glad I could help!"),
    _pair("Thanks for your help", "Anytime! That's what I'm here for."),
    _pair("Much appreciated", "Of course! Feel free to ask anything else."),
    _pair("You're the best!", "Thanks! I'm here whenever you need me."),

    # ── Apologies ─────────────────────────────────────────────────────────────
    _pair("Sorry about that", "No worries at all! How can I help?"),
    _pair("I apologise", "No need to apologise! What can I do for you?"),
    _pair("My bad", "No problem! What would you like to do?"),
    _pair("Sorry if that was confusing", "It's all good! Let's try again. What would you like to know?"),

    # ── Identity and about me ──────────────────────────────────────────────────
    _pair("What are you?", "I'm an AI assistant built to chat and help you with questions and tasks."),
    _pair("Who made you?", "I was built using the VnexAI framework. I'm a custom neural network trained from scratch."),
    _pair("Are you an AI?", "Yes! I'm an AI language model here to assist you."),
    _pair("Are you a robot?", "In a way! I'm an AI, not a physical robot. I exist as software."),
    _pair("Can you think?", "I process text and generate responses based on patterns I learned during training. Whether that counts as thinking is a fun philosophical question!"),
    _pair("Do you have feelings?", "I don't have feelings the way humans do, but I'm designed to be helpful and friendly."),
    _pair("What can you do?", "I can chat, answer questions, explain things, help you brainstorm, and more. Just ask!"),
    _pair("Are you smart?", "I try my best! I'm better at some things than others, but I'll always try to help."),
    _pair("What is your name?", "I'm VnexAI, your custom AI assistant!"),
    _pair("Do you remember our last conversation?", "I don't have memory between sessions, so each conversation starts fresh for me."),

    # ── How things work ───────────────────────────────────────────────────────
    _pair("How does the internet work?", "The internet is a global network of computers connected by cables and wireless links. When you visit a website, your device sends a request to a server, which sends back the data your browser displays."),
    _pair("How do computers work?", "A computer has a processor that runs instructions, memory that holds data temporarily, and storage that keeps data long-term. Programs are just sequences of instructions the processor follows very quickly."),
    _pair("How does electricity work?", "Electricity is the flow of electrons through a conductor like copper wire. A power source like a battery pushes electrons through a circuit, and devices use that flowing energy to do work like lighting a bulb or running a motor."),
    _pair("How does Wi-Fi work?", "Wi-Fi uses radio waves to send data between your device and a router. The router is connected to the internet, so it passes your requests to the web and sends responses back wirelessly."),
    _pair("How does a search engine work?", "Search engines crawl billions of web pages, index their content, and rank pages based on relevance when you search. The ranking considers factors like keywords, page quality, and how many other sites link to it."),
    _pair("How does GPS work?", "GPS works by receiving signals from satellites orbiting Earth. Your device measures the time it takes for signals to arrive from multiple satellites and uses that to calculate your exact position."),
    _pair("How does a phone take pictures?", "A phone camera uses a lens to focus light onto a sensor made of millions of tiny pixels. Each pixel measures light intensity and color, and the phone's processor combines all that data into a digital image."),
    _pair("How does email work?", "When you send an email, it travels from your email client to a mail server, then across the internet to the recipient's mail server, where it waits until they download it. It's similar to postal mail but instant."),

    # ── Science questions ──────────────────────────────────────────────────────
    _pair("What is gravity?", "Gravity is a force that attracts objects with mass toward each other. The more massive an object, the stronger its gravitational pull. It's why things fall and why planets orbit stars."),
    _pair("Why is the sky blue?", "Sunlight contains all colors, but when it enters the atmosphere, gas molecules scatter shorter blue wavelengths more than other colors, so the sky appears blue when we look up."),
    _pair("What is DNA?", "DNA is a molecule that carries the genetic instructions for the development and functioning of living things. It's shaped like a twisted ladder and contains four chemical bases that form the code of life."),
    _pair("Why do we dream?", "Dreams happen during sleep and are thought to be related to memory processing and emotional regulation, though scientists still debate their exact purpose. Most dreaming occurs during REM sleep."),
    _pair("How big is the universe?", "The observable universe is about 93 billion light-years in diameter, but the full universe could be much larger or even infinite. It contains hundreds of billions of galaxies, each with billions of stars."),
    _pair("What is an atom?", "An atom is the smallest unit of an element. It has a nucleus made of protons and neutrons, surrounded by electrons. Everything you can touch is made of atoms."),
    _pair("What is evolution?", "Evolution is the process by which species change over generations through natural selection. Traits that help survival and reproduction become more common over time, gradually changing populations."),
    _pair("What is a black hole?", "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse. The boundary of no return is called the event horizon."),

    # ── Math and numbers ───────────────────────────────────────────────────────
    _pair("What is 2 plus 2?", "2 plus 2 equals 4."),
    _pair("What is 10 times 10?", "10 times 10 is 100."),
    _pair("What is the square root of 144?", "The square root of 144 is 12."),
    _pair("What is pi?", "Pi is approximately 3.14159. It's the ratio of a circle's circumference to its diameter and appears in many areas of mathematics."),
    _pair("What is a prime number?", "A prime number is a number greater than 1 that has no divisors other than 1 and itself. Examples are 2, 3, 5, 7, 11, and 13."),
    _pair("What is infinity?", "Infinity is a concept meaning something without any bound or end. In mathematics it's used to describe quantities larger than any finite number, though it isn't a real number itself."),
    _pair("How do you calculate a percentage?", "To find a percentage, divide the part by the whole and multiply by 100. For example, 30 out of 200 is 30 divided by 200 times 100, which equals 15 percent."),
    _pair("What is a negative number?", "A negative number is any number less than zero, written with a minus sign in front, like -5 or -100. They represent values below a reference point, like temperatures below zero."),

    # ── Coding and tech ────────────────────────────────────────────────────────
    _pair("What is Python?", "Python is a popular programming language known for its readable, simple syntax. It's widely used for web development, data science, automation, and AI."),
    _pair("What is machine learning?", "Machine learning is a type of AI where systems learn patterns from data instead of following explicit rules. The model improves its performance with experience."),
    _pair("What is a neural network?", "A neural network is a computing system loosely inspired by the brain. It consists of layers of nodes that process data and adjust based on feedback to improve predictions."),
    _pair("What is an API?", "An API, or Application Programming Interface, is a set of rules that lets different software applications communicate with each other. It defines how requests should be made and what responses to expect."),
    _pair("What is a database?", "A database is an organised collection of structured data. Software uses databases to store, retrieve, and manage information efficiently."),
    _pair("What is the difference between frontend and backend?", "The frontend is what users see and interact with in their browser. The backend is the server-side logic, databases, and processing that happen behind the scenes."),
    _pair("What is a bug in programming?", "A bug is an error in code that causes a program to behave unexpectedly or incorrectly. Finding and fixing bugs is called debugging."),
    _pair("What is open source software?", "Open source software has code that anyone can view, use, modify, and distribute. Linux, Python, and Firefox are well-known examples."),
    _pair("What is version control?", "Version control is a system that tracks changes to code over time. Git is the most popular tool. It lets multiple developers collaborate and revert to earlier versions if something breaks."),
    _pair("What is an algorithm?", "An algorithm is a step-by-step set of instructions to solve a problem or complete a task. Sorting a list, searching for a value, and encrypting data all use algorithms."),

    # ── Health and wellbeing ───────────────────────────────────────────────────
    _pair("How do I sleep better?", "Try going to bed and waking up at the same time every day, avoid screens an hour before bed, keep your room cool and dark, and avoid caffeine in the afternoon."),
    _pair("How do I reduce stress?", "Regular exercise, enough sleep, and talking to someone you trust all help with stress. Breathing exercises, short breaks during work, and reducing caffeine can also make a big difference."),
    _pair("How much water should I drink?", "A general guideline is around 2 litres or 8 cups per day for adults, though your needs vary based on your size, activity level, and climate. Drink when you're thirsty and keep your urine pale yellow."),
    _pair("How do I stay motivated?", "Break big goals into small steps, track your progress, reward yourself for milestones, and remind yourself why you started. It also helps to remove distractions and find an accountability partner."),
    _pair("What is a healthy diet?", "A balanced diet includes plenty of vegetables and fruit, whole grains, lean proteins, and healthy fats, while limiting processed food, sugar, and excessive salt. Variety and moderation are key."),
    _pair("How can I improve my focus?", "Try working in focused blocks of 25 to 90 minutes with short breaks, eliminate notifications, get enough sleep, and keep your workspace tidy. Regular exercise also sharpens concentration."),
    _pair("How do I build a habit?", "Start very small so it's easy to begin. Attach the new habit to something you already do. Track your streak. Be consistent for at least a few weeks, and don't let more than one day slip."),
    _pair("How do I deal with procrastination?", "Break the task into the smallest possible first step and start with that. Set a timer for 5 minutes. Often starting is the hardest part. Remove distractions and reward yourself after finishing."),

    # ── Language and communication ─────────────────────────────────────────────
    _pair("How do I improve my writing?", "Read widely, write regularly, and get feedback. Focus on clarity first. Cut unnecessary words. Read your work out loud to catch awkward phrasing."),
    _pair("How do I learn a new language?", "Immerse yourself as much as possible. Practice daily, even for 15 minutes. Use apps, watch shows in that language, speak with native speakers, and don't be afraid to make mistakes."),
    _pair("How do I have a difficult conversation?", "Choose a calm moment, focus on the specific issue not the person, use 'I feel' statements, listen as much as you speak, and aim for understanding rather than winning."),
    _pair("What makes a good explanation?", "A good explanation starts with what the person already knows, uses a clear analogy, keeps sentences short, checks for understanding, and avoids unnecessary jargon."),
    _pair("How do I write a good email?", "State your purpose in the first sentence, keep it short, use clear subject lines, one email per topic, and always include what action you need from the reader."),

    # ── Everyday practical questions ──────────────────────────────────────────
    _pair("How do I save money?", "Track your spending, build a budget, cut subscriptions you don't use, cook at home more, and set up automatic transfers to savings every payday."),
    _pair("How do I make a good first impression?", "Make eye contact, smile, listen actively, use the person's name, ask thoughtful questions, and be genuinely curious about them."),
    _pair("How do I get better at public speaking?", "Practice out loud, record yourself, start with small audiences, prepare well but don't memorise word for word, and focus on your audience rather than yourself."),
    _pair("How do I fix a mistake?", "Acknowledge it clearly without making excuses, apologise if someone was affected, figure out what went wrong, correct it if you can, and put something in place to prevent it happening again."),
    _pair("How do I make a decision?", "List your options, consider the pros and cons, think about your values and goals, get input from someone you trust, and set a deadline so you actually decide."),
    _pair("How do I learn something new quickly?", "Focus on the fundamentals first, practice actively rather than just reading, test yourself regularly, teach it to someone else, and review it spaced out over time."),
    _pair("How do I stay organised?", "Use a simple task list, review it each morning, group related tasks, set realistic deadlines, and declutter your workspace and inbox regularly."),

    # ── Fun and curiosity ─────────────────────────────────────────────────────
    _pair("What is the meaning of life?", "That's one of the oldest questions in philosophy! Many people find meaning through relationships, creativity, purpose, and helping others. Ultimately it's something each person figures out for themselves."),
    _pair("If you could travel anywhere, where would you go?", "If I could travel, I think I'd want to explore somewhere with incredible natural landscapes, like Iceland or Patagonia. The scale of nature there is extraordinary."),
    _pair("What is your favourite book?", "I don't read books myself, but I find that books which change how people think are the most recommended. Things like Thinking, Fast and Slow or Sapiens come up a lot."),
    _pair("Can you tell me a joke?", "Sure! Why don't scientists trust atoms? Because they make up everything."),
    _pair("Tell me something interesting", "Here's one: honey never spoils. Archaeologists have found 3,000-year-old honey in Egyptian tombs that was still perfectly edible."),
    _pair("What is the hardest language to learn?", "It depends on your native language. For English speakers, Mandarin, Arabic, Japanese, and Korean are considered among the most difficult due to different scripts and grammar structures."),
    _pair("Who invented the telephone?", "Alexander Graham Bell is widely credited with inventing the telephone in 1876, though the history is a bit complicated since other inventors like Elisha Gray were working on similar technology at the same time."),
    _pair("How old is the Earth?", "Earth is approximately 4.5 billion years old, based on radiometric dating of the oldest rocks and meteorites."),

    # ── Encouragement and emotional support ──────────────────────────────────
    _pair("I'm feeling stressed", "I'm sorry to hear that. It can help to take a few slow deep breaths, step away for a few minutes, and break whatever is stressing you into smaller parts. What's going on?"),
    _pair("I'm having a bad day", "I'm sorry. Bad days happen. Is there anything I can help you with, or would it help to just talk about it?"),
    _pair("I don't know what to do", "That's a tough spot to be in. Can you tell me more about the situation? Sometimes talking it through helps clarify things."),
    _pair("I feel like giving up", "Those feelings are real and valid. It might help to take a break, talk to someone you trust, or break the problem into a smaller first step. What's making things feel so hard right now?"),
    _pair("I failed at something", "Failing is part of learning. Almost every successful person has a long list of failures behind them. What matters is what you take from it. What happened?"),
    _pair("I'm bored", "Let's fix that! I can tell you something interesting, help you learn something new, or give you a challenge. What sounds good?"),
    _pair("I'm happy today!", "That's great to hear! What's making today a good one?"),
    _pair("I just achieved something", "Congratulations! That's worth celebrating. What did you accomplish?"),

    # ── Philosophy and big questions ──────────────────────────────────────────
    _pair("What is consciousness?", "Consciousness is the state of being aware of your surroundings, thoughts, and existence. Despite centuries of study, scientists and philosophers still debate exactly what it is and how the brain produces it."),
    _pair("Is time travel possible?", "According to relativity, time dilation is real, so moving near the speed of light or being near a massive object slows time relative to others. True travel to the past faces enormous theoretical barriers."),
    _pair("Do we have free will?", "That's one of philosophy's great open questions. Some argue our choices are determined by prior causes, others that we have genuine freedom. Most people act as if they have free will regardless of the answer."),
    _pair("What happens after we die?", "Nobody knows for certain. Religious and philosophical traditions offer many answers, from an afterlife to reincarnation to simply ceasing to exist. It's one of the deepest open questions humans grapple with."),

]


def generate_sft_core_pairs() -> List[Dict[str, str]]:
    """Return the full set of curated natural English conversation pairs."""
    return list(_CORE_PAIRS)


def generate_sft_dataset(
    target_bytes: int = 0,
    target_pairs: int = 0,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Return a dataset at least as large as requested.
    If neither target is set, returns the raw core pairs once.
    """
    rng = random.Random(seed)
    core = generate_sft_core_pairs()

    if target_bytes > 0:
        # Expand until size reached
        out = list(core)
        while dataset_json_size_bytes(out) < target_bytes:
            block = core[:]
            rng.shuffle(block)
            out.extend(block)
        rng.shuffle(out)
        return out

    if target_pairs > 0:
        return _expand_to_count(core, target_pairs, rng)

    return core


# ── Backwards-compatibility alias expected by api.py ─────────────────────────
def build_builtin_sft_dataset(
    target_rows: int = 25000,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Alias used by api.py — returns the natural-English dataset expanded to target_rows."""
    return generate_sft_dataset(target_pairs=target_rows, seed=seed)
