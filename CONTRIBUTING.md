# 🤝 Contributing to JAX Bonsai

We love contributions! JAX Bonsai aims to be a community-driven collection of high-quality JAX NNX model implementations. Whether you're fixing a bug, adding a new model, improving documentation, or proposing new features, your help is greatly appreciated.

Please take a moment to review this document to understand how to contribute effectively.

---

## Ways to Contribute

There are many ways you can contribute to JAX Bonsai:

1.  **Reporting Bugs:** If you find a bug, please open an issue describing the problem clearly.
2.  **Suggesting Enhancements:** Have an idea for a new model, a better way to structure code, or a useful feature? Open an issue to discuss it.
3.  **Writing Code:**
    * **Fixing Bugs:** Submit a pull request with a fix for an existing bug.
    * **Adding New Models:** Implement a new state-of-the-art model using JAX NNX.
    * **Improving Existing Models:** Enhance performance, add features, or refactor existing model implementations.
    * **Writing Tests:** Improve code coverage by adding new tests.
4.  **Improving Documentation:** Enhance the `README.md` files, add clearer explanations, or create new guides.
5.  **Community Engagement:** Answer questions, help other users, and share your experiences.

---


## Contributing code using pull requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Sign the [Google Contributor License Agreement (CLA)](https://cla.developers.google.com/).
   For more information, see the Pull Request Checklist below.

2. Fork the JAX-bonsai repository by clicking the **Fork** button on the
   [repository page](http://www.github.com/jiyounha/jax-bonsai). This creates
   a copy of the JAX-bonsai repository in your own account.

3. `pip` installing your fork from source. This allows you to modify the code
   and immediately test it out:

   ```bash
   git clone https://github.com/YOUR_USERNAME/jax-bonsai
   cd jax-bonsai
   pip install -r build/test-requirements.txt  # Installs all testing requirements.
   ```

4. Add the JAX-bonsai repo as an upstream remote, so you can use it to sync your
   changes.

   ```bash
   git remote add upstream https://www.github.com/jiyounha/jax-bonsai
   ```

6. Create a branch where you will develop from:

   ```bash
   git checkout -b name-of-change
   ```

   And implement your changes using your favorite editor (we recommend
   [Visual Studio Code](https://code.visualstudio.com/)).

7. Once you are satisfied with your change, create a commit as follows (
   [how to write a commit message](https://chris.beams.io/posts/git-commit/)):

   ```bash
   git add file1.py file2.py ...
   git commit -m "Your commit message"
   ```

   Then sync your code with the main repo:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Finally, push your commit on your development branch and create a remote
   branch in your fork that you can use to create a pull request from:

   ```bash
   git push --set-upstream origin name-of-change
   ```

   Please ensure your contribution is a single commit (see {ref}`single-change-commits`)

10. Create a pull request from the JAX-bonsai repository and send it for review.
    Check the {ref}`pr-checklist` for considerations when preparing your PR, and
    consult [GitHub Help](https://help.github.com/articles/about-pull-requests/)
    if you need more information on using pull requests.
