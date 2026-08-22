/**
 * Test harness for .github/workflows/issue_validator.yml.
 *
 * GitHub Actions cannot be exercised locally, so the job scripts are extracted from the
 * workflow YAML (by the caller, which has a YAML parser) and run here against a stubbed
 * `github`/`context`. This runs the SHIPPED script text, not a copy of it - a mirror
 * would drift from the workflow and pin nothing.
 *
 * Usage: node issue_validator_harness.mjs <spec.json>
 *   spec: { restoreScript, validateScript, scenarios: [{ name, body, labels, latestRelease }] }
 *   out : JSON array of { name, restore: {...}, validate: {...} } on stdout
 */
import fs from 'node:fs';

const AsyncFn = Object.getPrototypeOf(async function () {}).constructor;

function makeStub({ body, labels, latestRelease }) {
  const state = { labels: [...labels], comments: [], addedLabels: [], removedLabels: [], logs: [] };
  const issue = {
    number: 1, body, state: 'open',
    user: { login: 'reporter', type: 'User' },
    get labels() { return state.labels.map(n => ({ name: n })); },
  };
  const github = {
    rest: {
      issues: {
        get: async () => ({ data: { ...issue, body, labels: state.labels.map(n => ({ name: n })) } }),
        addLabels: async ({ labels: ls }) => { state.addedLabels.push(...ls); state.labels.push(...ls); },
        removeLabel: async ({ name }) => { state.removedLabels.push(name); state.labels = state.labels.filter(l => l !== name); },
        createComment: async ({ body: b }) => { state.comments.push(b); },
        updateComment: async ({ body: b }) => { state.comments.push(b); },
        listComments: async () => ({ data: state.comments.map((b, i) => ({ user: { type: 'Bot' }, body: b, id: i })) }),
      },
      repos: {
        listReleases: async () => ({
          data: [{ tag_name: latestRelease, draft: false, prerelease: false }],
        }),
      },
    },
    paginate: async () => [],
  };
  const context = { eventName: 'issues', repo: { owner: 'o', repo: 'r' }, payload: { issue } };
  return { state, github, context };
}

async function run(src, stub) {
  const orig = console.log;
  console.log = (...a) => stub.state.logs.push(a.map(String).join(' '));
  try {
    await new AsyncFn('github', 'context', 'core', src)(stub.github, stub.context, {});
  } finally {
    console.log = orig;
  }
}

const spec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const sc of spec.scenarios) {
  const restore = makeStub(sc);
  await run(spec.restoreScript, restore);
  // The validate job re-reads the issue, so it sees whatever restore just applied.
  const validate = makeStub({ ...sc, labels: restore.state.labels });
  await run(spec.validateScript, validate);
  out.push({
    name: sc.name,
    restore: { addedLabels: restore.state.addedLabels, logs: restore.state.logs },
    validate: {
      addedLabels: validate.state.addedLabels,
      removedLabels: validate.state.removedLabels,
      comments: validate.state.comments,
      logs: validate.state.logs,
    },
  });
}
process.stdout.write(JSON.stringify(out, null, 2));
