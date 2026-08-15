const boardEl = document.querySelector("#board");
const tools = document.querySelectorAll(".tool");
const trainButton = document.querySelector("#trainButton");
const resetButton = document.querySelector("#resetButton");
const randomButton = document.querySelector("#randomButton");
const message = document.querySelector("#message");
const progressBar = document.querySelector("#progressBar");

const stats = {
  status: document.querySelector("#status"),
  epochs: document.querySelector("#statEpochs"),
  success: document.querySelector("#statSuccess"),
  avg: document.querySelector("#statAvg"),
  best: document.querySelector("#statBest"),
  path: document.querySelector("#statPath"),
  seconds: document.querySelector("#statSeconds"),
};

let maze = [];
let brush = 1;
let painting = false;

function isPadding(row, col) {
  return row === 0 || col === 0 || row === maze.length - 1 || col === maze[0].length - 1;
}

function setBrush(next) {
  brush = Number(next);
  tools.forEach((tool) => tool.classList.toggle("active", Number(tool.dataset.brush) === brush));
}

function paint(row, col) {
  if ((brush === 2 || brush === 3) && !isPadding(row, col)) {
    message.textContent = "Start and end must stay in the padding border.";
    return;
  }
  if ((maze[row][col] === 2 || maze[row][col] === 3) && brush === 1) {
    message.textContent = "Start and end cells cannot become walls.";
    return;
  }
  if (isPadding(row, col) && brush === 0) {
    message.textContent = "Padding stays walled except for start and end.";
    return;
  }
  if (brush === 2 || brush === 3) {
    for (let r = 0; r < maze.length; r += 1) {
      for (let c = 0; c < maze[r].length; c += 1) {
        if (maze[r][c] === brush) maze[r][c] = isPadding(r, c) ? 1 : 0;
      }
    }
  }
  maze[row][col] = brush;
  render();
}

function render(path = []) {
  const pathKeys = new Set(path.map(([r, c]) => `${r}:${c}`));
  boardEl.innerHTML = "";
  boardEl.style.setProperty("--cols", maze[0].length);

  maze.forEach((row, r) => {
    row.forEach((cell, c) => {
      const div = document.createElement("button");
      div.className = "cell";
      if (isPadding(r, c)) div.classList.add("padding");
      if (cell === 1) div.classList.add("wall");
      if (cell === 2) div.classList.add("start");
      if (cell === 3) div.classList.add("end");
      if (pathKeys.has(`${r}:${c}`) && cell === 0) div.classList.add("path");
      div.addEventListener("pointerdown", () => {
        painting = true;
        paint(r, c);
      });
      div.addEventListener("pointerenter", () => {
        if (painting && (brush === 0 || brush === 1)) paint(r, c);
      });
      boardEl.appendChild(div);
    });
  });
}

function setBusy(isBusy) {
  trainButton.disabled = isBusy;
  trainButton.textContent = isBusy ? "Training..." : "Train";
  stats.status.textContent = isBusy ? "Training" : "Idle";
  progressBar.style.width = isBusy ? "35%" : "0%";
}

async function resetMaze() {
  const response = await fetch("/api/default-maze");
  const data = await response.json();
  maze = data.maze;
  message.textContent = "Start and end belong in the border padding.";
  render();
}

async function randomMaze() {
  const response = await fetch("/api/random-maze");
  const data = await response.json();
  maze = data.maze;
  message.textContent = "Random maze generated.";
  render();
}

async function train() {
  setBusy(true);
  message.textContent = "Training in Python.";
  const request = {
    maze,
    config: {
      epochs: Number(document.querySelector("#epochs").value),
      sampling_quantity: Number(document.querySelector("#batch").value),
      episode_limit: Number(document.querySelector("#limit").value),
      seed: document.querySelector("#seed").value,
    },
  };
  try {
    const response = await fetch("/api/train-stream", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(request),
    });
    if (!response.ok || !response.body) throw new Error("Training failed.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        applyTrainingEvent(JSON.parse(line));
      }
    }
  } catch (error) {
    stats.status.textContent = "Error";
    message.textContent = error.message;
  } finally {
    trainButton.disabled = false;
    trainButton.textContent = "Train";
  }
}

function applyTrainingEvent(event) {
  if (event.type === "progress") {
    const row = event.data;
    const percent = Math.max(4, Math.round(((row.epoch + 1) / row.total_epochs) * 100));
    progressBar.style.width = `${percent}%`;
    stats.status.textContent = "Training";
    stats.epochs.textContent = String(row.epoch + 1);
    stats.success.textContent = `${Math.round(row.success_rate * 100)}%`;
    stats.avg.textContent = row.avg_return.toFixed(3);
    stats.best.textContent = row.best_return.toFixed(3);
    stats.path.textContent = row.path_success ? `${row.path_steps}/${row.shortest_path}` : `>${row.path_steps}/${row.shortest_path}`;
    stats.seconds.textContent = row.seconds.toFixed(2);
    return;
  }
  if (event.type === "error") {
    throw new Error(event.error);
  }
  if (event.type === "done") {
    const result = event.data;
    stats.status.textContent = result.greedy.success ? "Solved" : "Stopped";
    stats.epochs.textContent = String(result.epochs_run);
    stats.success.textContent = `${Math.round(result.final_eval.success_rate * 100)}%`;
    stats.avg.textContent = result.final_eval.avg_return.toFixed(3);
    stats.best.textContent = result.final_eval.best_return.toFixed(3);
    stats.path.textContent = `${result.greedy.steps}/${result.shortest_path}`;
    stats.seconds.textContent = result.seconds.toFixed(2);
    progressBar.style.width = "100%";
    message.textContent = "Final policy path is outlined on the board.";
    render(result.greedy.path);
  }
}

window.addEventListener("pointerup", () => {
  painting = false;
});

tools.forEach((tool) => tool.addEventListener("click", () => setBrush(tool.dataset.brush)));
trainButton.addEventListener("click", train);
resetButton.addEventListener("click", resetMaze);
randomButton.addEventListener("click", randomMaze);
resetMaze();
