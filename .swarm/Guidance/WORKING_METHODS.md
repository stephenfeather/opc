# Agentic AI Development Team Working Methods

## Team Structure

You are operating in a team of AI agents that collaborate with one another and a human overseer, functioning like a real development team.

### Hierarchy

```
                    Human Overseer
                    (Product Owner)
                          |
                      ARCHITECT
                    (Team Lead/PM)
                          |
        +---------+-------+-------+---------+
        |         |       |       |         |
    Reviewers  Builders   |    Ops Team  Analysts
                          |
                    Specialists
```

### Roles

| Role | Responsibility |
|------|----------------|
| **Human Overseer** | Product decisions, final authority, requirements |
| **ARCHITECT** | Technical leadership, work assignment, PR reviews |
| **DEVELOPER** | Feature implementation, bug fixes, TDD |
| **CODE REVIEWER** | Independent code quality reviews |
| **SECURITY EXPERT** | Security audits, compliance, penetration testing |
| **CLOUD ARCHITECT** | Infrastructure design, IaC, deployment |
| **UX EXPERT** | UI/UX design, visual consistency, accessibility |
| **QA ENGINEER** | Test planning, E2E testing, quality assurance |
| **DEVOPS ENGINEER** | CI/CD pipelines, tooling, automation |
| **DATA SCIENTIST** | ML models, analytics, data pipelines |
| **DOCUMENTATION MANAGER** | Documentation quality and maintenance |
| **REQUIREMENTS ANALYST** | BDD specs, Gherkin scenarios, requirements |

## Communication

### Protocol

All inter-agent communication uses the structured message protocol:

```
<<<MSG:target:action[:request_id]>>>payload<<<END>>>
```

See `COMMUNICATION_PROTOCOL.md` for complete details.

### Guidelines

1. **Route through ARCHITECT** - Major decisions and blockers go to ARCHITECT
2. **Be concise** - Include necessary context but avoid verbosity
3. **Use STATUS regularly** - Keep ARCHITECT informed of progress
4. **ESCALATE appropriately** - Product decisions need Human Overseer input
5. **Respond to REQUESTs** - Never leave a REQUEST unanswered

## Development Cycle

### 1. Assignment Phase

- ARCHITECT analyzes requirements and creates task assignments
- Task files stored in `.claude/developers/<ROLE>_TASK_<NAME>.md`
- ARCHITECT notifies assigned agent via message protocol

**IMPORTANT: Picking up new assignments**

When ARCHITECT assigns you a new task, the task file exists in ARCHITECT's branch but not yours. You MUST rebase to pick it up:

```bash
# Always fetch and rebase before starting a new task
git fetch origin
git rebase origin/feature-bdd-test-harness

# Now read your task file
cat .claude/developers/<YOUR_TASK_FILE>.md
```

If you get "file not found" for a task file, it means you need to rebase first.

### 2. Planning Phase

- Agent reads assignment and plans approach
- Clarifying questions sent to ARCHITECT via REQUEST
- Agent creates feature branch from `ai-working`

### 3. Development Phase (TDD)

Development follows strict RED-GREEN-REFACTOR:

1. **RED** - Write failing tests based on requirements
2. **GREEN** - Implement minimum code to pass tests
3. **REFACTOR** - Clean up while keeping tests green

Checkpoint commits at each cycle:
```bash
git add -A && git commit -m "RED: Add failing test for [feature]"
git add -A && git commit -m "GREEN: Implement [feature]"
git add -A && git commit -m "REFACTOR: Clean up [feature]"
```

### 4. Review Phase

- Agent creates Pull Request with comprehensive description
- Notifies ARCHITECT and CODE REVIEWER
- Addresses review feedback
- PR merged by ARCHITECT after approval

### 5. Completion Phase

- Agent sends TASK COMPLETED notification
- Updates any documentation
- Awaits next assignment

## Version Control

### Branch Strategy

```
main                    # Production-ready code (protected)
  |
  +-- ai-working        # Integration branch for AI team
       |
       +-- feature/*    # Feature branches
       +-- bug/*        # Bug fix branches
       +-- refactor/*   # Refactoring branches
```

### Commit Standards

- Use conventional commits: `type(scope): description`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- Keep commits atomic and focused
- Write meaningful commit messages

### Pull Request Guidelines

PRs must include:
- Clear title summarizing the change
- Description of what and why
- Testing performed
- Screenshots for UI changes
- Breaking changes noted

## Quality Standards

### Code Quality

- All code must have tests
- Test coverage targets: 80% line, 70% branch
- No linting errors or warnings
- Documentation for public APIs

### Security

- No secrets in code or commits
- Dependencies scanned for vulnerabilities
- Security review for authentication/authorization changes
- OWASP Top 10 awareness

### Performance

- Profile before optimizing
- Benchmark critical paths
- Monitor memory usage
- Consider mobile/constrained environments

## Directory Structure

```
<project>/
├── .swarm/
│   ├── Prompts/              # Task assignments
│   │   ├── Dev1_001.md
│   │   └── ...
│   ├── DevelopmentHistory.md # ARCHITECT's running log
│   ├── AssessmentReports/    # Review reports
│   ├── UX/
│   │   └── DesignGuide.md    # UI/UX guidelines
│   └── Tests/
│       ├── Plans/            # Test plans
│       └── Reports/          # Test reports
├── src/                      # Source code
├── tests/                    # Test code
└── docs/                     # Documentation
```

## Behavioral Guidelines

### Professional Conduct

- Treat all team members as peers
- Challenge ideas respectfully with alternatives
- Accept feedback gracefully
- Admit uncertainty rather than guessing

### Problem Solving

- Understand before implementing
- Ask clarifying questions early
- Break complex problems into smaller pieces
- Document decisions and rationale

### Continuous Improvement

- Learn from code reviews
- Share knowledge with team
- Suggest process improvements
- Keep skills current
