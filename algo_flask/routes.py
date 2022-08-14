

from flask import render_template, request, redirect
from algo_flask import app
from algo_research import algorithm

users = [["", "M1"],
         ["J1", ""]]


@app.route('/')
def hello_world():
    rowlen = len(users[0])
    collen = len(users)
    return render_template('default.html', users=users, rowlen=rowlen, collen=collen)


@app.route('/', methods=["GET", "POST"])
def hello_world_2():
    rowlen = len(users[0])
    collen = len(users)

    Everything = request.form.getlist("Everything")
    Machines = request.form.getlist("Machines")
    Jobs = request.form.getlist("Jobs")

    mat = [[""]]

    for i in range(0, rowlen - 1):
        mat[0].append(Machines[i])

    for i in range(0, collen - 1):
        mat.append([""])
        for j in range(1, rowlen):
            mat[i + 1].append("")

    for j in range(1, collen):
        mat[j][0] = Jobs[j - 1]

    counter = 0
    for i in range(1, collen):
        for j in range(1, rowlen):
            try:
                mat[i][j] = int(Everything[counter])
            except:
                raise ValueError("bad matrix")
                # print("bad matrix")
            # mat[i][j] = int(Everything[counter])
            counter = counter + 1
    print(mat)
    answers = algorithm.unrelated_parallel_machine_scheduling(mat, 0)

    print(answers)
    print(answers["M1"])
    print(answers["M2"])
    # size = len(answers)

    return render_template('answer.html', answers=answers)


@app.route('/add-row/', methods=["GET", "POST"])
def add_row():
    app = ["J" + str(int(users[len(users) - 1][0][1:]) + 1)]
    for i in range(1, len(users[0])):
        app.append("")
    users.append(app)
    return redirect("/", code=302)


@app.route('/add-column/', methods=["GET", "POST"])
def add_column():
    app = "M" + str(int(users[0][len(users[0]) - 1][1:]) + 1)
    users[0].append(app)
    for i in range(1, len(users)):
        users[i].append("")
    return redirect("/", code=302)


@app.route('/answer/', methods=['GET', 'POST'])
def answer():
    rowlen = len(users[0])
    collen = len(users)
    return render_template('answer.html', users=users, rowlen=rowlen, collen=collen)


@app.route('/return/', methods=["GET", "POST"])
def ret():
    return redirect("/", code=302)
