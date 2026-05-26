// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
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
