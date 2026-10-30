
const input = document.getElementById("prompt-input");
const output = document.getElementById("terminal-output");
const status = document.getElementById("status");
const clock = document.getElementById("clock");
const modelSelect = document.getElementById("model-select");

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
// Helpers
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

function getSelectedModel() {
    return modelSelect.value;
}

function getModelName(model) {
    return model === "sslm"
        ? "SSLM-100M"
        : "DumbassLM-135M";
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
models     Display available models
exit       Clear current input

Any other input will be sent to the selected model.`,
                "message-system"
            );
            return true;

        case "clear":
            output.innerHTML = "";
            return true;

        case "about":
            appendMessage(
                `SSLM TERMINAL

SSLM-100M
Architecture : Diagonal LTI State Space Model
Parameters  : Approximately 100M
Inference   : CPU

DumbassLM-135M
Architecture : Transformer (Llama-style)
Parameters  : 135M
Inference   : CPU

Backend : FastAPI`,
                "message-system"
            );
            return true;

        case "models":
            appendMessage(
                `AVAILABLE MODELS

[1] SSLM-100M
    Custom State Space Language Model

[2] DumbassLM-135M
    SmolLM2-135M-Instruct

Use the MODEL selector in the header to switch.`,
                "message-system"
            );
            return true;

        case "exit":
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
    modelSelect.disabled = true;

    const selectedModel = getSelectedModel();
    const modelName = getModelName(selectedModel);

    appendMessage(prompt, "message-prompt");

    const label = appendLabel(
        `${modelName} :: GENERATING RESPONSE...`
    );

    const responseElement = document.createElement("div");
    responseElement.className = "message message-answer";

    responseElement.textContent = "";

    output.appendChild(responseElement);

    setStatus(`${modelName} PROCESSING...`);
    scrollToBottom();

    try {

        const response = await fetch("/prompt", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                prompt: prompt,
                model: selectedModel,
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

        label.textContent = `${data.model_name} :: RESPONSE`;

        setStatus("SYSTEM READY");

    } catch (error) {

        responseElement.className = "message message-error";
        responseElement.textContent = `[ ERROR ] ${error.message}`;

        label.textContent = `${modelName} :: ERROR`;

        setStatus("REQUEST FAILED");

    } finally {

        isProcessing = false;

        input.disabled = false;
        modelSelect.disabled = false;

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
        input.style.height = "auto";

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
// Model selection
// --------------------------------------------------

modelSelect.addEventListener("change", () => {
    const modelName = getModelName(getSelectedModel());

    appendMessage(
        `[ SYSTEM ] Active model switched to ${modelName}.`,
        "message-system"
    );

    setStatus(`${modelName} SELECTED`);
    input.focus();
});


// --------------------------------------------------
// Initial state
// --------------------------------------------------

input.focus();
