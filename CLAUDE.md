
## Правило Виталия (29.10.2026): экономия GitHub Actions
- Полные тесты/CI запускать один раз на PR (pull_request: opened, synchronize, reopened). Не дублировать на push в рабочие ветки и повторно на push в main после merge.
- В каждом workflow: `concurrency: { group: ${{ github.workflow }}-${{ github.ref }}, cancel-in-progress: true }`.
- `paths-ignore` для `**/*.md`, `docs/**` и текстовых правок.
- Deploy-workflow после merge в main не трогать.
- iOS/macOS сборки — только по явной просьбе Виталия (workflow_dispatch).
- Изменения только через PR → CI → merge; ручные правки на сервере — только hotfix с немедленным PR.
