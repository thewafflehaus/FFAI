// FFAI — Fucking Fast Apple Inference
//
// See planning/plan.md and planning/architecture.md.

import Foundation

public enum FFAI {
    /// Library version, surfaced in CLI output + bench reports.
    /// **Bump this in lockstep with the git tag the release workflow
    /// will create** — see documentation/developing/publishing.md.
    /// Suffix with `-dev` on `dev` between releases so stale builds
    /// are easy to spot in logs.
    public static let version = "0.1.0"
}
