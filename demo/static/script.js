const form = document.getElementById("analysis-form");

const fileInput =
    document.getElementById("contract");

const browseButton =
    document.getElementById("browse-button");

const uploadBox =
    document.getElementById("upload-box");

const fileName =
    document.getElementById("file-name");

const analyzeButton =
    document.getElementById("analyze-button");

const buttonText =
    document.getElementById("button-text");

const loadingSpinner =
    document.getElementById("loading-spinner");

const answerSection =
    document.getElementById("answer-section");

const clausesSection =
    document.getElementById("clauses-section");

const errorSection =
    document.getElementById("error-section");

const errorMessage =
    document.getElementById("error-message");

const answer =
    document.getElementById("answer");

const clauses =
    document.getElementById("clauses");


// ============================================================
// File selection
// ============================================================

browseButton.addEventListener(
    "click",
    () => {
        fileInput.click();
    }
);


uploadBox.addEventListener(
    "click",
    (event) => {

        if (
            event.target !== browseButton
        ) {
            fileInput.click();
        }

    }
);


fileInput.addEventListener(
    "change",
    () => {

        if (fileInput.files.length > 0) {

            fileName.textContent =
                fileInput.files[0].name;

        } else {

            fileName.textContent = "";

        }

    }
);


// ============================================================
// Form submission
// ============================================================

form.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();

        // Hide previous errors/results
        hideError();

        answerSection.classList.add(
            "hidden"
        );

        clausesSection.classList.add(
            "hidden"
        );


        // ====================================================
        // Validate PDF
        // ====================================================

        if (
            fileInput.files.length === 0
        ) {

            showError(
                "Please upload a PDF contract."
            );

            return;
        }


        // ====================================================
        // Validate question
        // ====================================================

        const question =
            document
                .getElementById("question")
                .value
                .trim();


        if (!question) {

            showError(
                "Please enter a question."
            );

            return;
        }


        // ====================================================
        // Loading state
        // ====================================================

        analyzeButton.disabled = true;

        buttonText.textContent =
            "Analyzing...";

        loadingSpinner.classList.remove(
            "hidden"
        );


        // ====================================================
        // Prepare form data
        // ====================================================

        const formData =
            new FormData();

        formData.append(
            "contract",
            fileInput.files[0]
        );

        formData.append(
            "question",
            question
        );


        // ====================================================
        // Send request to Flask
        // ====================================================

        try {

            const response =
                await fetch(
                    "/analyze",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            // =================================================
            // Handle backend errors
            // =================================================

            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Analysis failed."
                );

            }


            // =================================================
            // Display answer
            // =================================================

            answer.textContent =
                data.answer;

            answerSection.classList.remove(
                "hidden"
            );


            // =================================================
            // Display supporting clauses
            // =================================================

            clauses.innerHTML = "";


            data.passages.forEach(
                (passage) => {

                    const clause =
                        document.createElement(
                            "div"
                        );


                    // ------------------------------------------------
                    // Highlight the highest-ranked passage
                    // ------------------------------------------------

                    if (
                        passage.rank === 1
                    ) {

                        clause.className =
                            "clause clause-primary";

                    } else {

                        clause.className =
                            "clause";

                    }


                    // ------------------------------------------------
                    // Clause header
                    // ------------------------------------------------

                    clause.innerHTML = `
                        <div class="clause-header">

                            <span class="clause-number">
                                Excerpt ${passage.rank}
                            </span>

                            ${
                                passage.rank === 1
                                    ? `
                                        <span class="clause-score">
                                            Most relevant
                                        </span>
                                      `
                                    : ""
                            }

                        </div>

                        <div class="clause-text"></div>
                    `;


                    // ------------------------------------------------
                    // Insert clause text safely
                    // ------------------------------------------------

                    clause
                        .querySelector(
                            ".clause-text"
                        )
                        .textContent =
                        passage.text;


                    clauses.appendChild(
                        clause
                    );

                }
            );


            clausesSection.classList.remove(
                "hidden"
            );


            // =================================================
            // Scroll to answer
            // =================================================

            answerSection.scrollIntoView({
                behavior: "smooth"
            });


        } catch (error) {

            console.error(
                "Analysis error:",
                error
            );


            showError(
                error.message ||
                "Something went wrong while analyzing the contract."
            );

        } finally {

            // =================================================
            // Restore button
            // =================================================

            analyzeButton.disabled =
                false;

            buttonText.textContent =
                "Analyze Contract";

            loadingSpinner.classList.add(
                "hidden"
            );

        }

    }
);


// ============================================================
// Error handling
// ============================================================

function showError(message) {

    errorMessage.textContent =
        message;

    errorSection.classList.remove(
        "hidden"
    );

    errorSection.scrollIntoView({
        behavior: "smooth"
    });

}


function hideError() {

    errorSection.classList.add(
        "hidden"
    );

    errorMessage.textContent = "";

}