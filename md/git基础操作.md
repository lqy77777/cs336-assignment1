# Git 基础操作：从克隆到上传更新

本文整理本次对话涉及的 Git 操作。示例默认在仓库根目录执行，分支名为 `main`。如果你的分支名不同，请替换为实际名称。

## 1. 先理解几个概念

| 概念 | 含义 |
| --- | --- |
| Git | 在本地记录文件修改历史的版本管理工具 |
| GitHub | 托管远程 Git 仓库的平台 |
| 本地仓库 | 电脑上的项目及其 Git 历史，历史和配置通常保存在隐藏的 `.git/` 目录中 |
| 远程仓库 | GitHub 等服务器上的仓库 |
| 工作区 | 你平时查看和编辑文件的地方 |
| 暂存区 | 选择这次准备提交的修改的地方 |
| 提交（commit） | 保存到本地 Git 历史中的一次修改记录 |
| 分支（branch） | 一条开发线，`main` 是常见的分支名 |

日常上传的过程是：

```text
编辑文件 → git add → git commit → git push
           暂存修改    本地提交      上传提交
```

**保存文件不等于提交，提交也不等于上传。** Git 不会自动把你每次保存的内容同步到 GitHub。

## 2. 查看当前目录与仓库状态

```bash
pwd
ls -la
git status
git branch --show-current
git remote -v
```

| 命令 | 用途 |
| --- | --- |
| `pwd` | 查看当前所在目录 |
| `ls -la` | 查看目录内容，包括隐藏文件 |
| `git status` | 查看当前分支、已暂存及未暂存的修改、未跟踪文件 |
| `git branch --show-current` | 查看当前分支名 |
| `git remote -v` | 查看远程仓库的别名及地址 |

可以先问自己：我在哪个仓库、哪个分支？这次准备上传哪些文件？上传目标是谁？

## 3. 克隆仓库

### 克隆到一个新子文件夹

```bash
git clone https://github.com/stanford-cs336/assignment1-basics.git
cd assignment1-basics
```

也可以指定本地文件夹名称：

```bash
git clone https://github.com/stanford-cs336/assignment1-basics.git my-assignment1
cd my-assignment1
```

`my-assignment1` 只是本地文件夹名，不会改变 GitHub 上的仓库名称。

### 克隆到当前文件夹

```bash
git clone https://github.com/stanford-cs336/assignment1-basics.git .
```

末尾的 `.` 表示当前目录。此方式要求当前目录为空，否则会报错：

```text
fatal: destination path '.' already exists and is not an empty directory.
```

用 `ls -la` 检查：即使 Finder 看起来没有文件，也可能存在 `.DS_Store` 等隐藏文件。`AGENTS.md` 等普通文件也会使目录不为空。

如果确认目录里只有 `.DS_Store`，可以删除这个 macOS 文件夹显示设置文件，再克隆：

```bash
rm .DS_Store
git clone https://github.com/stanford-cs336/assignment1-basics.git .
```

若还有其他文件，可以选择克隆到子文件夹。不要为了克隆而随意删除已有项目文件。

终端命令中的仓库地址应使用纯网址，不要粘贴 Markdown 的 `[网址](网址)` 格式。

## 4. origin 和 upstream 是什么

它们都是远程仓库地址的别名，不是分支名，也不是固定权限或固定角色。

`git clone` 默认把克隆来源命名为 `origin`。因此，刚克隆 Stanford 仓库时，`origin` 指向的是 Stanford。

把项目关联到自己的仓库后，常见约定是：

| 别名 | 常见指向 | 用途 |
| --- | --- | --- |
| `origin` | 自己的 GitHub 仓库 | 上传自己的提交 |
| `upstream` | 原作者的仓库 | 后续获取原项目的更新 |

`git remote -v` 输出中的 `(fetch)` 表示获取时使用的地址，`(push)` 表示推送时使用的地址。同一个地址出现两行，并不代表有两个仓库，也不代表你拥有向该地址推送的权限。

编写本文时，本项目的实际配置为：

```text
origin → https://github.com/lqy77777/cs336-assignment1.git
```

目前没有 `upstream`。下面的配置命令用于理解和按需设置，不需要在已经配置好的仓库里全部重做。

## 5. 把克隆的项目关联到自己的新仓库

克隆后可以继续修改，再推送到自己的新仓库；原来的提交历史仍会保留。

先在 GitHub 创建一个空仓库。不要初始化 README、`.gitignore` 或许可证，以便直接推送已有的本地历史。

如果 `origin` 仍指向原作者，先改名，再添加自己的地址：

```bash
git remote rename origin upstream
git remote add origin https://github.com/你的用户名/新仓库名.git
```

把示例网址换成自己的实际仓库地址。第一条命令只修改本地别名，不会给 GitHub 上的仓库改名。

如果已经有 `origin`，只是想更换它指向的地址，可以使用：

```bash
git remote set-url origin https://github.com/你的用户名/新仓库名.git
```

检查配置，然后首次推送：

```bash
git remote -v
git push -u origin main
```

`-u` 会建立本地 `main` 与远程 `origin/main` 的跟踪关系，方便以后直接执行 `git push`。

这里的“分支跟踪关系”有时也叫 upstream，与名为 `upstream` 的远程别名是两个不同概念。

## 6. 直接从命令行创建 GitHub 仓库

GitHub CLI 的命令是 `gh`，它可以操作 GitHub 上的仓库。你当前的电脑已安装该工具。

先检查登录状态：

```bash
gh auth status
```

如果尚未登录，执行并按提示操作：

```bash
gh auth login
```

要从已有的本地仓库创建一个新的 GitHub 仓库，可以使用：

```bash
gh repo create my-cs336-assignment1 --private --source=. --remote=origin --push
```

| 参数 | 含义 |
| --- | --- |
| `my-cs336-assignment1` | 新仓库名，默认创建在当前登录账号下 |
| `--private` | 创建私有仓库；公开仓库使用 `--public` |
| `--source=.` | 使用当前本地仓库 |
| `--remote=origin` | 将新仓库关联为 `origin` |
| `--push` | 将本地已有提交上传到新仓库 |

**此命令用于创建尚不存在的新仓库，且示例要求 `origin` 名称尚未被占用。** 如果它仍指向原作者，可以先按上一节将其改名。如果你已经关联好自己的仓库，直接按下一节上传即可。

`--push` 不会替你提交工作区里尚未提交的修改。参数说明可查阅 [GitHub CLI 官方文档](https://cli.github.com/manual/gh_repo_create)。

## 7. 日常修改后如何上传更新

### 第一步：查看并选择修改

```bash
git status
git diff
```

`git diff` 显示已跟踪文件中尚未暂存的修改，不显示未跟踪文件的内容。

暂存指定文件，例如：

```bash
git add AGENTS.md
```

如果确认当前目录及其子目录下的全部变化都应该提交，也可以执行：

```bash
git add .
```

在仓库根目录执行时，这会暂存未被忽略的新文件、修改和删除。初学时可以优先指定文件，避免把 `.DS_Store` 等无关文件一起提交。

如果某个已跟踪文件被你删除，`git add 文件路径` 也可以暂存它的删除操作。

### 第二步：检查并提交

```bash
git diff --cached
git commit -m "更新协作约定"
```

`git diff --cached` 查看已经暂存、准备进入这次提交的修改。提交说明应简要描述实际做了什么。

`git commit` 只记录已暂存的内容。如果暂存后又修改了同一文件，需要再次执行 `git add`，才能把后续修改也放进此次提交。

### 第三步：上传

```bash
git push origin main
```

如果已经用 `git push -u origin main` 建立跟踪关系，且当前就在 `main`，通常可以简写为：

```bash
git push
```

上传后运行 `git status` 检查本地状态，并刷新自己的 GitHub 仓库页面确认内容。

### 示例：上传本文

在仓库根目录执行：

```bash
git add md/git基础操作.md
git diff --cached
git commit -m "添加 Git 基础操作笔记"
git push origin main
```

提交会包含暂存区中所有已暂存的修改，所以提交前需要确认 `git diff --cached` 中的内容符合预期。

## 8. 删除或恢复 upstream

如果不需要保留原仓库地址，可以执行：

```bash
git remote remove upstream
git remote -v
```

这会删除本地的远程关联及相关远程跟踪信息，不会删除原作者的 GitHub 仓库，也不会删除本地工作区文件或影响 `origin`。

以后如果需要重新关联：

```bash
git remote add upstream https://github.com/stanford-cs336/assignment1-basics.git
```

## 9. 常见提示怎么理解

| 提示 | 含义与下一步 |
| --- | --- |
| `destination path '.' ... is not an empty directory` | 当前目录不为空；用 `ls -la` 检查，或克隆到子文件夹 |
| `remote origin already exists` | 已经有名为 `origin` 的远程；先用 `git remote -v` 查看，再按需改名或更换地址 |
| `No such remote: 'upstream'` | 当前没有这个远程别名；如果你的目标是移除它，就无需重复操作 |
| `nothing to commit, working tree clean` | 没有新的本地修改需要提交；不代表所有已有提交都上传了 |
| `Everything up-to-date` | 此次推送涉及的提交已在远程；未提交的文件修改不会被上传 |
| `current branch ... has no upstream branch` | 尚未设置分支跟踪关系；当前分支为 `main` 时可执行 `git push -u origin main` |

## 10. 用几个问题检查理解

1. 修改并保存文件后，直接执行 `git push` 会上传该修改吗？
   - 不会，需要先暂存并提交。
2. 本地文件夹改名后，GitHub 仓库会跟着改名吗？
   - 不会，本地文件夹名和远程仓库名相互独立。
3. 删除 `upstream` 会删除原作者的代码仓库吗？
   - 不会，只会移除本地对它的关联。
4. 怎么确认自己的提交将被推送到哪里？
   - 用 `git remote -v` 查看地址，再用 `git push origin main` 明确指定远程和分支。
