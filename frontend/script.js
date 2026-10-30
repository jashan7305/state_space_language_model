
const input = document.getElementById("prompt-input");
const output = document.getElementById("terminal-output");
const status = document.getElementById("status");
const clock = document.getElementById("clock");

let isProcessing = false;


// --------------------------------------------------
// Clock
// --------------------------------------------------

function updateClock() {
    clock.textContent = new Date().toLocaleTimeString("en-GB");
}

setInterval(updateClock, 1000);
updateClock();


// --------------------------------------------------
// Terminal output helpers
// --------------------------------------------------

function scrollToBottom() {
    output.scrollTop = output.scrollHeight;
}

function appendMessage(content, className) {
    const message = document.createElement("div");
    message.className = `message ${className}`;
    message.textContent = content;

    output.appendChild(message);
    scrollToBottom();

    return message;
}

function appendLabel(content) {
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = content;

    output.appendChild(label);
    return label;
}

function setStatus(message) {
    status.textContent = message;
}


// --------------------------------------------------
// Terminal commands
// --------------------------------------------------

function executeCommand(command) {
    const normalized = command.trim().toLowerCase();

    switch (normalized) {

        case "help":
            appendMessage(
                `AVAILABLE COMMANDS

help       Display available commands
clear      Clear terminal screen
about      Model information
exit       Clear current input

Any other input will be sent to SSLM.`,
                "message-system"
            );
            return true;

        case "clear":
            output.innerHTML = "";
            return true;

        case "about":
            appendMessage(
                `SSLM :: STATE SPACE LANGUAGE MODEL

Architecture : Diagonal LTI State Space Model
Parameters   : Approximately 100M class
Inference    : CPU
Backend      : FastAPI
Interface    : Terminal

A custom language model trained from scratch.`,
                "message-system"
            );
            return true;

        case "exit":
            input.value = "";
            appendMessage("Session remains active.", "message-system");
            return true;

        default:
            return false;
    }
}


// --------------------------------------------------
// API request
// --------------------------------------------------

async function sendPrompt(prompt) {

    if (isProcessing || !prompt.trim()) return;

    isProcessing = true;
    input.disabled = true;

    appendMessage(prompt, "message-prompt");

    const label = appendLabel("SSLM :: GENERATING RESPONSE...");
    const responseElement = document.createElement("div");

    responseElement.className = "message message-answer";
    responseElement.textContent = "";

    output.appendChild(responseElement);

    setStatus("PROCESSING REQUEST...");

    try {

        const response = await fetch("/prompt", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                prompt: prompt,
                max_new_tokens: 150,
                temperature: 0.8,
                top_k: 40
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Request failed.");
        }

        responseElement.textContent = data.answer;
        label.textContent = "SSLM :: RESPONSE";

        setStatus("SYSTEM READY");

    } catch (error) {

        responseElement.className = "message message-error";
        responseElement.textContent = `[ ERROR ] ${error.message}`;

        label.textContent = "SSLM :: ERROR";

        setStatus("REQUEST FAILED");

    } finally {

        isProcessing = false;
        input.disabled = false;
        input.focus();

        scrollToBottom();
    }
}


// --------------------------------------------------
// Input handling
// --------------------------------------------------

input.addEventListener("keydown", async (event) => {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        if (isProcessing) return;

        const prompt = input.value.trim();

        if (!prompt) return;

        input.value = "";

        if (executeCommand(prompt)) {
            setStatus("SYSTEM READY");
            input.focus();
            return;
        }

        await sendPrompt(prompt);
    }
});


// --------------------------------------------------
// Auto-resize input
// --------------------------------------------------

input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight}px`;
});


// --------------------------------------------------
// Initial focus
// --------------------------------------------------

input.focus();
