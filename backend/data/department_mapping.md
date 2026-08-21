# RTI Department / PIO Mapping

Curated lookup used by the RTI Drafting Agent (Step 13/14) to infer which
department or Public Information Officer (PIO) an RTI application should be
addressed to, based on the subject matter of the user's request.

This is a starting set covering common, high-frequency RTI topics. It is
NOT exhaustive — India has PIOs at the central, state, and local body level
for nearly every government function. Where a request doesn't clearly map
to one of these categories, the agent should leave the department field as
a placeholder for the user to fill in rather than guessing.

Format: one entry per topic area, with example request types and the
department/PIO to address.

## Public Hospital / Health Services
- Example requests: staff vacancies, hospital equipment status, medicine stock shortage, treatment records
- Department: State/District Health Department — District Medical Officer / Chief Medical Officer

---

## Ration Card / Public Distribution System
- Example requests: ration card application status, PDS shop irregularities, rejection of ration card
- Department: Department of Food, Civil Supplies and Consumer Affairs (state-level) — District Food & Supplies Officer

## Land Records / Property
- Example requests: land record mutation status, land dispute documents, property tax assessment
- Department: Revenue Department (state-level) — Tehsildar / District Collector's Office

## Birth / Death / Marriage Certificates
- Example requests: delayed birth certificate, correction in death certificate, marriage registration status
- Department: Municipal Corporation / Gram Panchayat — Registrar of Births and Deaths

## Police / FIR Matters
- Example requests: FIR copy, status of police complaint, action taken report
- Department: State Police Department — Superintendent of Police / Station House Officer (SHO) of the concerned police station

## Pension (old-age, widow, disability)
- Example requests: pension application status, delayed pension disbursal
- Department: Department of Social Justice and Empowerment (state-level) — District Social Welfare Officer

## Employment Guarantee Scheme (MGNREGA)
- Example requests: wage payment delay, job card issues, muster roll records
- Department: District Rural Development Agency (state-level) — Programme Officer, MGNREGA

## Passport
- Example requests: passport application delay, police verification status
- Department: Ministry of External Affairs — Regional Passport Office

## Income Tax
- Example requests: refund status, assessment order copy
- Department: Central Board of Direct Taxes (CBDT) — Income Tax Office (jurisdictional)

## Education / School Admissions
- Example requests: RTE admission quota compliance, school recognition status, exam result discrepancy
- Department: Department of School Education (state-level) — District Education Officer

## Municipal Services (water, sanitation, roads)
- Example requests: water connection delay, garbage collection complaint, road repair status
- Department: Municipal Corporation / Municipal Council — concerned Ward Office

## Electricity
- Example requests: connection delay, billing dispute, meter complaint
- Department: State Electricity Distribution Company (DISCOM) — concerned Sub-Division Office

## Consumer Court / Consumer Affairs
- Example requests: status of a filed consumer complaint, information on consumer forum proceedings
- Department: Department of Consumer Affairs — District Consumer Disputes Redressal Commission

## Central Government Employment / Recruitment
- Example requests: exam result details, recruitment process status, reservation roster
- Department: Union Public Service Commission (UPSC) or concerned recruiting Ministry/Department — Central Public Information Officer (CPIO)

---

## Notes for the drafting agent
- If a request spans multiple categories (e.g. a ration card AND a pension issue), draft two separate RTI applications rather than combining them — RTI applications should generally be specific to one department.
- If no category matches, leave `[Department / Public Authority Name]` and `[PIO Designation]` as explicit placeholders in the draft rather than guessing — an RTI sent to the wrong PIO is often transferred or rejected, so a wrong guess is worse than an honest placeholder.
- State-level department names vary slightly by state (this file uses common/generic naming) — the draft should note "designation and exact department name may vary by state; confirm before submitting" when using this mapping.
