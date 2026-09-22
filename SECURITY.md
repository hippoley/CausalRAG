# Security Policy

## Scope

CausalRAG sits on the execution boundary between model proposals and real tools, so security issues can have consequences beyond incorrect text.

Please treat the following as security-sensitive:

- bypassing runtime validity or temporal guards;
- executing a candidate that was rejected by the runtime;
- leaking provider credentials to the browser;
- bypassing owner access for external model use;
- unsafe defaults for side-effecting tools;
- accepting human/operator messages as trusted evidence without an explicit contract;
- trace or replay data exposing secrets;
- counterfactual replay mutating the live environment.

## Reporting

Please do not publish exploit details in a public issue before a fix path exists.

Use GitHub's private vulnerability reporting feature when available. Include:

- affected version / commit;
- minimal reproduction;
- expected vs actual execution boundary;
- whether a real external side effect can occur;
- whether credentials, trace content, or operator context can leak.

## Deployment boundary

The built-in workbench is a research and development surface, not an authorization system.

For public deployments:

- keep provider credentials server-side;
- enable `BRANCHPOINT_REQUIRE_PROBE_AUTH=true`;
- set a strong `BRANCHPOINT_PROBE_ACCESS_TOKEN`;
- enable secure cookies behind HTTPS;
- treat application IAM and business authorization as separate controls;
- do not expose side-effecting tools that the host application would not independently permit.

## Supported versions

Security fixes target the latest version on the default branch unless a release note says otherwise.
