// Sokoban solver for make.fun sokoban-solver-01.
//
// Strategy
// --------
// Generated levels are a one-cell-wide hallway with 1-4 rooms opening off it.
// A box pushed out of its room through the door can never come back, so in a
// push-optimal solution every box stays inside its own room and the level's
// optimum P* is the sum of the per-room optima. This solver therefore:
//
//   1. Decomposes the floor graph into rooms. A room is a component that
//      splits off the remaining region when one articulation cell of the
//      floor graph is removed, and that holds equally many boxes and goals
//      (more than zero). Rooms are peeled off one at a time; whatever is
//      left (hallway plus the last room) is one final sub-problem.
//   2. Solves each sub-problem with a push-level A* that is optimal for that
//      room: states are (sorted box cells, canonical player cell), the
//      heuristic is the sum over boxes of the static push distance to the
//      nearest room goal (reverse pull-BFS, other boxes ignored), static
//      dead squares and frozen 2x2 blocks are pruned, boxes belonging to
//      other sub-problems are treated as walls, and push targets are
//      confined to the room plus a one-cell halo around it.
//   3. Stitches the per-room move strings together by replaying each room's
//      solution on the global board to update the player and box positions,
//      then verifies that every box ends on a goal. If decomposition or a
//      room solve fails, it falls back to a whole-board push A*.
//
// The search is bounded primarily by expansion counts and every tie is
// broken by fixed orders; a generous wall-clock safety deadline per level
// only stops a pathological search from hanging the run (that level is then
// answered empty), it does not shape normal solutions.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed direction order used everywhere (determinism): up, down, left, right.
const DIRS = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

// Work budgets. Expansion counts are the primary, deterministic budget; the
// wall-clock deadlines are only a safety net so a pathological level can
// never hang or OOM the run — when a budget is hit the search stops, that
// level is answered empty, and the program moves on to the next level.
const ROOM_EXPANSION_LIMIT = 600000;
const WHOLE_EXPANSION_LIMIT = 200000;
const PROGRAM_START = Date.now();
const LEVEL_BUDGET_MS = 20000;
const PROGRAM_BUDGET_MS = 50000;

// A generated room holds 3-4 boxes inside a 6x6 slot, plus its door cell.
const MIN_ROOM_BOXES = 3;
const MAX_ROOM_BOXES = 4;
const MAX_ROOM_CELLS = 37;

const DEBUG = !!process.env.SOKOBAN_DEBUG;
const dbg = (...args) => {
  if (DEBUG) console.error(...args);
};

// ---------------------------------------------------------------------------
// Level parsing
// ---------------------------------------------------------------------------

function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const n = w * h;
  const wall = new Array(n).fill(true);
  const goals = [];
  const goalSet = new Set();
  let boxes = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    const row = rows[y];
    for (let x = 0; x < w; x++) {
      const ch = x < row.length ? row[x] : '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = false;
      if (ch === '.' || ch === '*' || ch === '+') {
        goals.push(i);
        goalSet.add(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  // Floor neighbours per cell (fixed direction order).
  const nbrs = new Array(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const list = [];
      if (!wall[i]) {
        for (let d = 0; d < 4; d++) {
          const nx = x + DIRS[d].dx;
          const ny = y + DIRS[d].dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const j = ny * w + nx;
          if (!wall[j]) list.push({ to: j, dir: d });
        }
      }
      nbrs[i] = list;
    }
  }
  return { w, h, n, wall, goals, goalSet, boxes, player, nbrs };
}

// ---------------------------------------------------------------------------
// Static push distances: for every cell, the minimum number of pushes that
// brings a lone box from that cell to one of `targetGoals`, moving only
// through `allowed` cells, with the pushing player standing on floor that is
// not blocked by `staticBlocked`. Computed by reverse (pull) BFS from the
// goals. Infinity marks a static dead square.
// ---------------------------------------------------------------------------

function pushDistances(level, targetGoals, allowed, staticBlocked) {
  const { w, n, wall } = level;
  const dist = new Array(n).fill(Infinity);
  const queue = [];
  for (const g of targetGoals) {
    if (allowed.has(g) && dist[g] === Infinity) {
      dist[g] = 0;
      queue.push(g);
    }
  }
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    const cx = cur % w;
    const cy = (cur / w) | 0;
    for (let d = 0; d < 4; d++) {
      // A box arrived at `cur` by a push in direction d, so before the push
      // it sat at prev = cur - d and the player stood at stand = cur - 2d.
      const px = cx - DIRS[d].dx;
      const py = cy - DIRS[d].dy;
      const sx = cx - 2 * DIRS[d].dx;
      const sy = cy - 2 * DIRS[d].dy;
      if (px < 0 || py < 0 || px >= w || sy < 0 || sx < 0 || sx >= w) continue;
      if (py >= level.h || sy >= level.h) continue;
      const prev = py * w + px;
      const stand = sy * w + sx;
      if (wall[prev] || wall[stand]) continue;
      if (!allowed.has(prev)) continue;
      if (staticBlocked.has(stand)) continue;
      if (dist[prev] <= dist[cur] + 1) continue;
      dist[prev] = dist[cur] + 1;
      queue.push(prev);
    }
  }
  return dist;
}

// ---------------------------------------------------------------------------
// Player reachability: BFS over floor avoiding `blocked` (walls are already
// excluded by the neighbour table; blocked holds box cells). Returns the
// parent map for path reconstruction and the canonical (smallest) cell of
// the reachable region.
// ---------------------------------------------------------------------------

function playerReach(level, start, blocked) {
  const parent = new Map();
  const parentDir = new Map();
  parent.set(start, -1);
  const queue = [start];
  let head = 0;
  let canonical = start;
  while (head < queue.length) {
    const cur = queue[head++];
    if (cur < canonical) canonical = cur;
    for (const { to, dir } of level.nbrs[cur]) {
      if (blocked.has(to) || parent.has(to)) continue;
      parent.set(to, cur);
      parentDir.set(to, dir);
      queue.push(to);
    }
  }
  return { parent, parentDir, canonical };
}

function walkString(reach, target) {
  // Rebuild the walk from the BFS root to `target` as move letters.
  let out = '';
  let cur = target;
  while (reach.parent.get(cur) !== -1) {
    const d = reach.parentDir.get(cur);
    out = DIRS[d].ch + out;
    cur = reach.parent.get(cur);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Frozen 2x2 check: true when some 2x2 square containing `cell` consists
// only of walls and boxes and holds a box that is not on a goal.
// ---------------------------------------------------------------------------

function makesFrozenBlock(level, cell, boxSet) {
  const { w, h, wall, goalSet } = level;
  const cx = cell % w;
  const cy = (cell / w) | 0;
  for (const ox of [-1, 0]) {
    for (const oy of [-1, 0]) {
      const x0 = cx + ox;
      const y0 = cy + oy;
      if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) continue;
      let allBlocked = true;
      let hasUngoledBox = false;
      for (const dx of [0, 1]) {
        for (const dy of [0, 1]) {
          const c = (y0 + dy) * w + (x0 + dx);
          if (wall[c]) continue;
          if (boxSet.has(c)) {
            if (!goalSet.has(c)) hasUngoledBox = true;
            continue;
          }
          allBlocked = false;
        }
      }
      if (allBlocked && hasUngoledBox) return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Push-level A* for one sub-problem.
//
//   subBoxes:  sorted starting cells of this room's boxes
//   subGoals:  this room's goal cells
//   otherBoxes: cells of every other box (treated as walls)
//   allowed:   cells a box of this room may be pushed onto (room + halo)
//   playerStart: the player's current global cell
//
// Returns the move string (walks in lower case, pushes in upper case) that
// puts every sub-box on a sub-goal with the fewest pushes, or null.
// ---------------------------------------------------------------------------

function solveSubProblem(level, subBoxes, subGoals, otherBoxes, allowed, playerStart, expansionLimit, deadline) {
  const goalSet = new Set(subGoals);
  const otherSet = new Set(otherBoxes);
  const dist = pushDistances(level, subGoals, allowed, otherSet);

  const heuristic = (boxes) => {
    let sum = 0;
    for (const b of boxes) {
      const d = dist[b];
      if (d === Infinity) return Infinity;
      sum += d;
    }
    return sum;
  };

  const startBoxes = [...subBoxes].sort((a, b) => a - b);
  if (heuristic(startBoxes) === Infinity) return null;
  if (startBoxes.every((b) => goalSet.has(b))) return '';

  // Node store; node 0 is the start.
  const nodeBoxes = [startBoxes];
  const nodePlayer = [playerStart]; // actual player cell after arriving here
  const nodeG = [0];
  const nodeParent = [-1];
  const nodeSegment = ['']; // moves from the parent node to this node
  const best = new Map(); // state key -> best g seen

  const stateKey = (boxes, canonical) => boxes.join(',') + '|' + canonical;

  const startReach = playerReach(level, playerStart, new Set([...startBoxes, ...otherSet]));
  best.set(stateKey(startBoxes, startReach.canonical), 0);

  // Binary heap ordered by (f asc, g desc, insertion order asc).
  const heap = [];
  let counter = 0;
  const entryLess = (a, b) =>
    a.f < b.f || (a.f === b.f && (a.g > b.g || (a.g === b.g && a.order < b.order)));
  const heapPush = (e) => {
    heap.push(e);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!entryLess(heap[i], heap[p])) break;
      const t = heap[i];
      heap[i] = heap[p];
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && entryLess(heap[c + 1], heap[c])) c++;
        if (!entryLess(heap[c], heap[i])) break;
        const t = heap[i];
        heap[i] = heap[c];
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  heapPush({ f: heuristic(startBoxes), g: 0, node: 0, order: counter++ });

  let expanded = 0;
  while (heap.length > 0) {
    const cur = heapPop();
    const boxes = nodeBoxes[cur.node];
    const g = nodeG[cur.node];
    if (cur.g !== g) continue;
    if (boxes.every((b) => goalSet.has(b))) {
      // Reconstruct the move string from the parent chain.
      let out = '';
      for (let t = cur.node; t > 0; t = nodeParent[t]) out = nodeSegment[t] + out;
      return out;
    }
    if (expanded >= expansionLimit) return null;
    if (expanded % 1024 === 0 && Date.now() > deadline) return null;
    expanded++;

    const boxSet = new Set(boxes);
    const blocked = new Set([...boxSet, ...otherSet]);
    const reach = playerReach(level, nodePlayer[cur.node], blocked);
    const { w } = level;

    for (let bi = 0; bi < boxes.length; bi++) {
      const box = boxes[bi];
      const bx = box % w;
      const by = (box / w) | 0;
      for (let d = 0; d < 4; d++) {
        const standX = bx - DIRS[d].dx;
        const standY = by - DIRS[d].dy;
        const destX = bx + DIRS[d].dx;
        const destY = by + DIRS[d].dy;
        if (standX < 0 || standY < 0 || destX < 0 || destY < 0) continue;
        if (standX >= w || destX >= w || standY >= level.h || destY >= level.h) continue;
        const stand = standY * w + standX;
        const dest = destY * w + destX;
        if (!reach.parent.has(stand)) continue; // player cannot get behind the box
        if (level.wall[dest] || boxSet.has(dest) || otherSet.has(dest)) continue;
        if (!allowed.has(dest)) continue; // boxes stay in their room (+ halo)
        if (dist[dest] === Infinity) continue; // static dead square
        const nextBoxes = boxes.slice();
        nextBoxes[bi] = dest;
        const nextSet = new Set(nextBoxes);
        if (makesFrozenBlock(level, dest, new Set([...nextSet, ...otherSet]))) continue;
        nextBoxes.sort((a, b) => a - b);
        const ng = g + 1;
        const hVal = heuristic(nextBoxes);
        if (hVal === Infinity) continue;
        // The player ends the push standing on the box's old cell.
        const nextReach = playerReach(level, box, new Set([...nextSet, ...otherSet]));
        const key = stateKey(nextBoxes, nextReach.canonical);
        const seen = best.get(key);
        if (seen !== undefined && seen <= ng) continue;
        best.set(key, ng);
        const idx = nodeBoxes.length;
        nodeBoxes.push(nextBoxes);
        nodePlayer.push(box);
        nodeG.push(ng);
        nodeParent.push(cur.node);
        nodeSegment.push(walkString(reach, stand) + DIRS[d].ch.toUpperCase());
        heapPush({ f: ng + hVal, g: ng, node: idx, order: counter++ });
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Room decomposition by articulation points of the floor graph.
// ---------------------------------------------------------------------------

function articulationPoints(level, region) {
  // Tarjan's algorithm over the subgraph induced by `region`.
  const disc = new Map();
  const low = new Map();
  const parent = new Map();
  const points = new Set();
  let time = 0;
  const dfs = (u) => {
    disc.set(u, time);
    low.set(u, time);
    time++;
    let children = 0;
    for (const { to } of level.nbrs[u]) {
      if (!region.has(to)) continue;
      if (!disc.has(to)) {
        parent.set(to, u);
        children++;
        dfs(to);
        low.set(u, Math.min(low.get(u), low.get(to)));
        if (parent.get(u) === undefined && children > 1) points.add(u);
        if (parent.get(u) !== undefined && low.get(to) >= disc.get(u)) points.add(u);
      } else if (to !== parent.get(u)) {
        low.set(u, Math.min(low.get(u), disc.get(to)));
      }
    }
  };
  for (const cell of [...region].sort((a, b) => a - b)) {
    if (!disc.has(cell)) {
      parent.set(cell, undefined);
      dfs(cell);
    }
  }
  return points;
}

function componentsWithout(level, region, removed) {
  const seen = new Set([removed]);
  const comps = [];
  for (const start of [...region].sort((a, b) => a - b)) {
    if (seen.has(start)) continue;
    const comp = new Set();
    const queue = [start];
    seen.add(start);
    let head = 0;
    while (head < queue.length) {
      const cur = queue[head++];
      comp.add(cur);
      for (const { to } of level.nbrs[cur]) {
        if (!region.has(to) || seen.has(to)) continue;
        seen.add(to);
        queue.push(to);
      }
    }
    comps.push(comp);
  }
  return comps;
}

function decompose(level) {
  const floor = new Set();
  for (let i = 0; i < level.n; i++) if (!level.wall[i]) floor.add(i);
  const remaining = new Set(floor);
  const rooms = [];
  for (;;) {
    const points = articulationPoints(level, remaining);
    let chosen = null; // { comp, boxCount, minCell }
    const consider = (comp) => {
      if (comp.has(level.player)) return;
      let boxCount = 0;
      let goalCount = 0;
      let minCell = Infinity;
      for (const c of comp) {
        if (c < minCell) minCell = c;
        if (level.boxes.includes(c)) boxCount++;
        if (level.goalSet.has(c)) goalCount++;
      }
      if (boxCount !== goalCount) return;
      // A single generated room holds 3-4 boxes in a 6x6 slot (+ its door).
      // This also rejects the complement trick: the rest of the board minus
      // a small hallway piece balances too, but it is far too big and holds
      // far too many boxes to be one room.
      if (boxCount < MIN_ROOM_BOXES || boxCount > MAX_ROOM_BOXES) return;
      if (comp.size > MAX_ROOM_CELLS) return;
      if (
        chosen === null ||
        boxCount > chosen.boxCount ||
        (boxCount === chosen.boxCount &&
          (comp.size < chosen.comp.size ||
            (comp.size === chosen.comp.size && minCell < chosen.minCell)))
      ) {
        chosen = { comp, boxCount, minCell };
      }
    };
    for (const a of [...points].sort((x, y) => x - y)) {
      for (const comp of componentsWithout(level, remaining, a)) {
        const before = chosen;
        consider(comp);
        if (chosen === before) {
          // The component alone does not balance, but the articulation cell
          // itself may hold the room's last box (a box sitting in the door)
          // or its last goal: then the room is the component plus that cell.
          const extended = new Set(comp);
          extended.add(a);
          consider(extended);
        }
      }
    }
    if (chosen === null) break;
    rooms.push(chosen.comp);
    for (const c of chosen.comp) remaining.delete(c);
  }
  return { rooms, remaining };
}

// ---------------------------------------------------------------------------
// Replay: apply a move string to (player, boxes); returns the new state or
// null when a move is illegal.
// ---------------------------------------------------------------------------

function replay(level, player, boxes, moves) {
  const boxSet = new Set(boxes);
  let p = player;
  const { w } = level;
  for (const ch of moves) {
    const lower = ch.toLowerCase();
    const d = DIRS.findIndex((dir) => dir.ch === lower);
    if (d < 0) return null;
    const px = p % w;
    const py = (p / w) | 0;
    const nx = px + DIRS[d].dx;
    const ny = py + DIRS[d].dy;
    if (nx < 0 || ny < 0 || nx >= w || ny >= level.h) return null;
    const next = ny * w + nx;
    if (level.wall[next]) return null;
    if (boxSet.has(next)) {
      const bx = nx + DIRS[d].dx;
      const by = ny + DIRS[d].dy;
      if (bx < 0 || by < 0 || bx >= w || by >= level.h) return null;
      const beyond = by * w + bx;
      if (level.wall[beyond] || boxSet.has(beyond)) return null;
      boxSet.delete(next);
      boxSet.add(beyond);
    }
    p = next;
  }
  return { player: p, boxes: [...boxSet].sort((a, b) => a - b) };
}

function withHalo(level, region) {
  const allowed = new Set(region);
  for (const c of region) {
    for (const { to } of level.nbrs[c]) allowed.add(to);
  }
  return allowed;
}

// ---------------------------------------------------------------------------
// Whole-level solve: decompose, solve room by room, stitch, verify; fall
// back to a whole-board push A* when anything fails.
// ---------------------------------------------------------------------------

function solveLevel(text) {
  const level = parseLevel(text);
  const deadline = Math.min(Date.now() + LEVEL_BUDGET_MS, PROGRAM_START + PROGRAM_BUDGET_MS);
  const { rooms, remaining } = decompose(level);
  const regions = [...rooms, remaining];
  dbg(
    'level: rooms peeled =',
    rooms.length,
    'region sizes =',
    regions.map((r) => r.size).join('/'),
  );

  let player = level.player;
  let boxes = [...level.boxes];
  let solution = '';
  let ok = true;

  for (const region of regions) {
    const subBoxes = boxes.filter((b) => region.has(b));
    const subGoals = level.goals.filter((g) => region.has(g));
    if (subBoxes.length === 0) continue;
    dbg('  sub-problem: cells =', region.size, 'boxes =', subBoxes.length, 'goals =', subGoals.length);
    if (subBoxes.length !== subGoals.length) {
      ok = false;
      break;
    }
    if (subBoxes.every((b) => level.goalSet.has(b))) continue;
    const otherBoxes = boxes.filter((b) => !region.has(b));
    const segment = solveSubProblem(
      level,
      subBoxes,
      subGoals,
      otherBoxes,
      withHalo(level, region),
      player,
      ROOM_EXPANSION_LIMIT,
      deadline,
    );
    dbg('  sub-problem result:', segment === null ? 'FAILED' : `solved, ${segment.length} moves`);
    if (segment === null) {
      ok = false;
      break;
    }
    const after = replay(level, player, boxes, segment);
    if (after === null) {
      ok = false;
      break;
    }
    solution += segment;
    player = after.player;
    boxes = after.boxes;
  }

  if (ok && boxes.every((b) => level.goalSet.has(b))) return solution;

  // Fallback: one push A* over the whole board, under a tight expansion cap.
  dbg('  falling back to whole-board search (boxes =', level.boxes.length, ')');
  const allFloor = new Set();
  for (let i = 0; i < level.n; i++) if (!level.wall[i]) allFloor.add(i);
  const whole = solveSubProblem(
    level,
    level.boxes,
    level.goals,
    [],
    allFloor,
    level.player,
    WHOLE_EXPANSION_LIMIT,
    deadline,
  );
  dbg('  fallback result:', whole === null ? 'FAILED' : `solved, ${whole.length} moves`);
  return whole ?? '';
}

// ---------------------------------------------------------------------------
// Entry point: answer every instance in one run.
// ---------------------------------------------------------------------------

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
