# GitHub Professional Repository Standards
> A Complete Reference Guide for IDE Agents & Developers

---

## 1. Repository Structure

A clean folder hierarchy signals engineering maturity. Separate concerns from day one and maintain naming consistency throughout the project lifecycle.

- Organize into logical top-level directories: `src/`, `docs/`, `tests/`, `assets/`, `config/`
- Remove all temporary files, debug artifacts, log files, and duplicate assets before pushing
- Keep file and folder names lowercase with hyphens (kebab-case), e.g., `user-auth/`, `api-client.ts`
- Never mix frontend, backend, and infrastructure code in a single flat directory

**Recommended top-level structure:**

```
my-project/
├── src/              # Application source code
│   ├── components/   # (frontend) UI components
│   ├── routes/       # API route handlers
│   └── utils/        # Shared utilities
├── tests/            # Unit, integration, and e2e tests
├── docs/             # Architecture, API, and design docs
├── .env.example      # Environment variable template
├── .gitignore        # Correctly configured
└── README.md         # Professional documentation
```

---

## 2. README.md Requirements

The README is the front door of your project. It must sell the project, onboard contributors, and answer every common question without requiring the reader to dig through source code.

### 2.1 Required Sections

- Project title with one-line description
- Screenshot or demo GIF (placed near the top, above the fold)
- Live demo link (if deployed)
- Feature list — be specific, not vague (e.g., "JWT-based auth with refresh tokens", not "authentication")
- Full tech stack with version numbers where relevant
- Step-by-step installation and local setup
- Environment variable documentation (reference `.env.example`)
- Usage instructions with example commands or screenshots
- Folder structure overview
- Known limitations and future improvements
- Engineering decisions and challenges overcome

### 2.2 Quality Bar

| ❌ Avoid | ✅ Prefer |
|---------|---------|
| `## Installation`<br>`Clone and run npm install then start.` | `## Installation`<br>` ```bash`<br>`git clone https://github.com/you/app`<br>`cd app && npm install`<br>`cp .env.example .env`<br>`npm run dev  # http://localhost:3000`<br>` ``` ` |

---

## 3. Code Quality Standards

Readable, modular, consistently formatted code is non-negotiable. Code that only the author can understand is a liability.

- Functions and components must do one thing — keep them under ~50 lines; split if larger
- Variable and function names must be descriptive: `fetchUserProfile()`, not `getData()`
- Remove all commented-out dead code; use version control to recover old code
- Zero `console.log()` calls in production code unless explicitly necessary with a comment
- Run your linter (ESLint, Prettier, Black, etc.) before every push — zero warnings
- Never hardcode secrets, URLs, or environment-specific values inline

| ❌ Avoid | ✅ Prefer |
|---------|---------|
| `const x = await fetch('/api/u/' + id);`<br>`console.log(x);`<br>`// const z = await oldFetch(id);` | `const user = await fetchUserById(userId);`<br>`// Removed legacy endpoint in v2.1` |

---

## 4. Commit Message Standards

Every commit is a permanent record. Use Conventional Commits format so history is scannable, changelogs can be generated automatically, and collaborators understand context without reading diffs.

### 4.1 Format

```
<type>(<scope>): <short imperative description>

[optional body — explain WHY, not what]

[optional footer: BREAKING CHANGE or issue reference]
```

### 4.2 Allowed Types

| Type | When to use |
|------|-------------|
| `feat` | A new feature or capability added to the application |
| `fix` | A bug fix — reference the issue number in the footer |
| `refactor` | Code restructured without changing external behaviour |
| `docs` | Documentation only changes (README, comments, guides) |
| `test` | Adding or updating tests — no production code change |
| `chore` | Tooling, dependencies, CI config, build scripts |
| `perf` | Performance improvement with measurable impact |
| `style` | Formatting only — no logic change (whitespace, semicolons) |

### 4.3 Examples

| ❌ Avoid | ✅ Prefer |
|---------|---------|
| `update`<br>`final`<br>`done`<br>`fix`<br>`add stuff`<br>`changes` | `feat(auth): add JWT refresh token rotation`<br>`fix(api): resolve null pointer in user lookup`<br>`docs: add architecture diagram to README`<br>`chore: upgrade eslint to v9` |

---

## 5. Deployment Requirements

A repository without a live demo is significantly less impactful. Deploy every major project before the final push.

- **Frontend:** deploy to Vercel or Netlify (automatic GitHub integration recommended)
- **Backend:** deploy to Render, Railway, Fly.io, or similar — ensure the service does not sleep on first request if possible
- Verify the live demo URL is responsive and all major flows work before linking in the README
- Add a status badge to the README (build passing, deployment live)
- Use environment variables for all deployment-specific config — never hardcode staging vs production values

---

## 6. Testing Requirements

Untested code pushed to GitHub signals that a project is not ready for professional scrutiny. Run the full test suite locally before every push.

- `npm test` / `pytest` / `cargo test` must pass with zero failures
- `npm run build` (or equivalent) must succeed cleanly
- TypeScript: `tsc --noEmit` must produce zero errors
- ESLint / Prettier: zero errors, zero warnings
- Manually verify every user-facing feature and API endpoint
- For critical projects, add a CI pipeline (GitHub Actions) that blocks merges on test failure

---

## 7. Documentation Standards

Documentation lives alongside code. Undocumented architecture and undocumented APIs have to be reverse-engineered by the next developer — that developer might be you in six months.

- Add an `ARCHITECTURE.md` for any project with more than two services or layers
- Document all API endpoints — method, path, request body, response schema, error codes
- Include ER diagrams for any project with a non-trivial database schema
- Document all environment variables in `.env.example` with a comment explaining each one
- Non-obvious code logic must have inline comments explaining WHY, not what

---

## 8. Security Requirements

A single leaked secret committed to a public repository can compromise infrastructure permanently. Git history is not a safe place for credentials — even if you delete the file later, the commit remains.

- Use `.env` files for all secrets — never commit them
- Provide a `.env.example` with placeholder values and comments; commit that instead
- Verify `.gitignore` includes: `.env`, `node_modules/`, `dist/`, `__pycache__/`, `*.log`
- Run `git status` before every commit and visually scan the diff for accidental secret inclusion
- Consider using `git-secrets` or a pre-commit hook to block secret patterns automatically
- Review all authentication and authorization paths for logic flaws before publishing

| ❌ Avoid | ✅ Prefer |
|---------|---------|
| `# .env (committed)`<br>`DATABASE_URL=postgres://admin:pass@prod.db/app`<br>`JWT_SECRET=supersecret` | `# .env.example (committed)`<br>`DATABASE_URL=postgres://user:password@localhost:5432/dbname`<br>`JWT_SECRET=your-256-bit-secret-here` |

---

## 9. Repository Presentation

Your GitHub profile is your portfolio. How repositories are presented is as important as their content.

- Pin your 4–6 strongest, most complete projects on your GitHub profile
- Repository name: kebab-case, descriptive, no version numbers (`task-manager`, not `my-app-v2-final`)
- Repository description (shown in search results): one clear sentence under 120 characters
- Add relevant topic tags: e.g., `react`, `typescript`, `rest-api`, `postgresql` — improves discoverability
- Set the **website** field to the live demo URL if deployed
- Keep the default branch named `main`, not `master`

---

## 10. Professional Standards

The sum of these practices defines how seriously you take your craft. Quality over quantity — five outstanding repositories are worth more than twenty mediocre ones.

- Every repository must solve a real, identifiable problem or demonstrate a clear skill
- Prefer depth over breadth: a fully documented, tested, deployed project beats five half-finished ones
- Maintain the same engineering bar across all repositories — inconsistency undermines credibility
- Revisit older repositories periodically: update dependencies, fix broken links, improve documentation

---

## 11. Final Verification Checklist

Complete every item before marking a repository as ready to publish. This checklist is a gate, not a suggestion.

| ☐ | Item | Acceptance Criteria |
|---|------|---------------------|
| ☐ | **README complete** | All required sections present; screenshots or GIF included |
| ☐ | **Live demo working** | URL loads, core flows verified, link in README and repo settings |
| ☐ | **No secrets exposed** | `git log` diff reviewed; `.env` not committed; `.gitignore` correct |
| ☐ | **Build successful** | `npm run build` / `cargo build` / `python -m build` exits with 0 |
| ☐ | **Tests passing** | Full test suite: zero failures, zero skipped (unless documented) |
| ☐ | **Linting clean** | ESLint / Prettier / Black: zero errors, zero warnings |
| ☐ | **TypeScript clean** | `tsc --noEmit`: zero errors |
| ☐ | **Environment documented** | `.env.example` committed with comments for every variable |
| ☐ | **Code formatted** | Consistent style applied; no dead code or stray `console.log` |
| ☐ | **Commit history clean** | Conventional Commits format; no "update" or "final" messages |
| ☐ | **Repo description set** | One-sentence description and relevant topic tags added on GitHub |
| ☐ | **Architecture docs** | `ARCHITECTURE.md` present for multi-service or complex projects |

---

> **Do not push incomplete, broken, undocumented, or untested code.**
> Every repository is a permanent, public reflection of your engineering standards.
