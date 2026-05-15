vision

I want to create a pipeline that ingests paper pdfs and extracts information on field as indicated in base-data/field and paper list.xlsx

How the extraction looks like is provided in the csv.

You may process the pdfs for easier parsing.

I want information extracted for fields that apply to the whole paper which includes stuff like author, title, intervention etc

If information is not available for any of the fields, they should be clearly marked as "data not available". Only extract from what is available in the file.


After that I want to identify tables that have "effect sizes"

Effect sizes are described as any quantitative measure such as difference in means, regression coefficient that provides an estimate of the difference between comparison groups.
In the absence of an effect size, we can also capture means and standard deviations of the comparison groups.

Effect sizes should be extracted.

I want the extraction to run on my local machine.

the data should be stored in a google sheet.

The data should then be available to review users to view.

This should be done on a web app with several users with their own IDs and passwords.
Users will either confirm or modify or delete or add to what the pipeline extracts.
They should be able to do that for each effect size.

The users can also note if a re-extraction is needed.